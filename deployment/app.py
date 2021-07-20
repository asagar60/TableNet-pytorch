import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytesseract
from io import StringIO


pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

TRANSFORM = A.Compose([
                #ToTensor --> Normalize(mean, std)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value = 255,
                ),
                ToTensorV2()
            ])

class DenseNet(nn.Module):
    def __init__(self, pretrained = True, requires_grad = True):
        super(DenseNet, self).__init__()
        denseNet = torchvision.models.densenet121(pretrained=True).features
        self.densenet_out_1 = torch.nn.Sequential()
        self.densenet_out_2 = torch.nn.Sequential()
        self.densenet_out_3 = torch.nn.Sequential()

        for x in range(8):
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8,10):
            self.densenet_out_2.add_module(str(x), denseNet[x])

        self.densenet_out_3.add_module(str(10), denseNet[10])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):

        out_1 = self.densenet_out_1(x) #torch.Size([1, 256, 64, 64])
        out_2 = self.densenet_out_2(out_1) #torch.Size([1, 512, 32, 32])
        out_3 = self.densenet_out_3(out_2) #torch.Size([1, 1024, 32, 32])
        return out_1, out_2, out_3

class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
                        in_channels = 256,
                        out_channels = 256,
                        kernel_size = kernels[0],
                        stride = strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(
                        in_channels = 256,
                        out_channels=128,
                        kernel_size = kernels[1],
                        stride = strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(
                        in_channels = 128 + channels[0],
                        out_channels = 256,
                        kernel_size = kernels[2],
                        stride = strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(
                        in_channels = 256 + channels[1],
                        out_channels = 1,
                        kernel_size = kernels[3],
                        stride = strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  #[1, 256, 32, 32]
        out = self.upsample_1_table(x) #[1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1) #[1, 640, 64, 64]
        out = self.upsample_2_table(out) #[1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1) #[1, 512, 128, 128]
        out = self.upsample_3_table(out) #[1, 3, 1024, 1024]
        return out

class ColumnDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(ColumnDecoder, self).__init__()
        self.conv_8_column = nn.Sequential(
                        nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = kernels[0], stride = strides[0]),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.8),
                        nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = kernels[0], stride = strides[0])
                        )
        self.upsample_1_column = nn.ConvTranspose2d(
                        in_channels = 256,
                        out_channels=128,
                        kernel_size = kernels[1],
                        stride = strides[1])
        self.upsample_2_column = nn.ConvTranspose2d(
                        in_channels = 128 + channels[0],
                        out_channels = 256,
                        kernel_size = kernels[2],
                        stride = strides[2])
        self.upsample_3_column = nn.ConvTranspose2d(
                        in_channels = 256 + channels[1],
                        out_channels = 1,
                        kernel_size = kernels[3],
                        stride = strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_8_column(x)  #[1, 256, 32, 32]
        out = self.upsample_1_column(x) #[1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1) #[1, 640, 64, 64]
        out = self.upsample_2_column(out) #[1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1) #[1, 512, 128, 128]
        out = self.upsample_3_column(out) #[1, 3, 1024, 1024]
        return out

class TableNet(nn.Module):
    def __init__(self):
        super(TableNet, self).__init__()

        self.base_model = DenseNet(pretrained = False, requires_grad = True)
        self.pool_channels = [512, 256]
        self.in_channels = 1024
        self.kernels = [(1,1), (1,1), (2,2),(16,16)]
        self.strides = [(1,1), (1,1), (2,2),(16,16)]

        #common layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8))

        self.table_decoder = TableDecoder(self.pool_channels, self.kernels, self.strides)
        self.column_decoder = ColumnDecoder(self.pool_channels, self.kernels, self.strides)

    def forward(self, x):

        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out) #[1, 256, 32, 32]
        table_out = self.table_decoder(conv_out, pool_3_out, pool_4_out) #torch.Size([1, 1, 1024, 1024])
        column_out = self.column_decoder(conv_out, pool_3_out, pool_4_out) #torch.Size([1, 1, 1024, 1024])
        return table_out,column_out

@st.cache(allow_output_mutation=True)
def load_model():

    model = TableNet()
    model.load_state_dict(torch.load("densenet_config_4_model_checkpoint.pth.tar")['state_dict'])
    model.eval()
    return model

def predict(img_path):
    with st.spinner('Processing..'):
        orig_image = Image.open(img_path).resize((1024, 1024))
        test_img = np.array(orig_image.convert('LA').convert("RGB"))

        now = datetime.now()
        image = TRANSFORM(image = test_img)["image"]
        with torch.no_grad():
            image = image.unsqueeze(0)
            #with torch.cuda.amp.autocast():
            table_out, _  = model(image)
            table_out = torch.sigmoid(table_out)

        #remove gradients
        table_out = (table_out.detach().numpy().squeeze(0).transpose(1,2,0) > 0.5).astype(np.uint8)

        #get contours of the mask to get number of tables
        contours, table_heirarchy = cv2.findContours(table_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        table_contours = []
        #ref: https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
        #remove bad contours
        for c in contours:

            if cv2.contourArea(c) > 3000:
                table_contours.append(c)

        if len(table_contours) == 0:
            st.write("No Table detected")

        table_boundRect = [None]*len(table_contours)
        for i, c in enumerate(table_contours):
            polygon = cv2.approxPolyDP(c, 3, True)
            table_boundRect[i] = cv2.boundingRect(polygon)

        #table bounding Box
        table_boundRect.sort()

        orig_image = np.array(orig_image)
        #draw bounding boxes
        color = (0,0,255)
        thickness = 4

        for x,y,w,h in table_boundRect:
            cv2.rectangle(orig_image, (x,y),(x+w,y+h), color, thickness)

        st.image(orig_image)

        end_time = datetime.now()
        difference = end_time - now
        #print("Total Time : {} seconds".format(difference))
        time = "{}".format(difference)

        st.write(f"{time} secs")

        st.write("Predicted Tables")

        image = test_img[...,0].reshape(1024, 1024).astype(np.uint8)

        for i,(x,y,w,h) in enumerate(table_boundRect):
            image_crop = image[y:y+h,x:x+w]
            data = pytesseract.image_to_string(image_crop)
            try:
                df = pd.read_csv(StringIO(data),sep=r'\|',lineterminator=r'\n',engine='python')
                st.write(f" ## Table {i+1}")
                st.write(df)
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(StringIO(data),delim_whitespace=True,lineterminator=r'\n',engine='python')
                    st.write(f" ## Table {i+1}")
                    st.write(df)
                except pd.errors.ParserError:
                    st.write(f" ## Table {i+1}")
                    st.write(data)



with st.spinner("Loading Last Checkpoint"):
    model = load_model()

st.header("Data Extraction from Tables")
#upload files
file = st.file_uploader("Please upload an Image file")

if file is not None:
    predict(file)
