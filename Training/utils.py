import torch
import random
import numpy as np
import os
import pandas as pd
import config
import cv2
import matplotlib.pyplot as plt
from dataset import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A 
from albumentations.pytorch import ToTensorV2

TRANSFORM = A.Compose([
                #ToTensor --> Normalize(mean, std) 
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value = 255,
                ),
                ToTensorV2()
            ])


def seed_all(SEED_VALUE= config.SEED):
    
    random.seed(SEED_VALUE)
    os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_data_loaders(data_path = config.DATAPATH):
    df = pd.read_csv(data_path)
    train_data, test_data  = train_test_split(df, test_size = 0.2, random_state = config.SEED, stratify = df.hasTable)

    train_dataset = ImageFolder(train_data, isTrain = True, transform = None)
    test_dataset = ImageFolder(test_data, isTrain = False, transform = None)

    train_loader =  DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle=True, num_workers = 4, pin_memory=True)
    test_loader =  DataLoader(test_dataset, batch_size = 8, shuffle=False, num_workers = 4, pin_memory=True)

    return train_loader, test_loader

#Checkpoint
def save_checkpoint(state, filename = "model_checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint Saved at ",filename)

def load_checkpoint(checkpoint, model, optimizer = None):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    tr_metrics = checkpoint['train_metrics']
    te_metrics = checkpoint['test_metrics']
    return last_epoch, tr_metrics, te_metrics

def write_summary(writer, tr_metrics, te_metrics, epoch):

    writer.add_scalar("Table loss/Train", tr_metrics['table_loss'], global_step=epoch)
    writer.add_scalar("Table loss/Test", te_metrics['table_loss'], global_step=epoch)
    
    writer.add_scalar("Table Acc/Train", tr_metrics['table_acc'], global_step=epoch)
    writer.add_scalar("Table Acc/Test", te_metrics['table_acc'], global_step=epoch)

    writer.add_scalar("Table F1/Train", tr_metrics['table_f1'], global_step=epoch)
    writer.add_scalar("Table F1/Test", te_metrics['table_f1'], global_step=epoch)

    writer.add_scalar("Table Precision/Train", tr_metrics['table_precision'], global_step=epoch)
    writer.add_scalar("Table Precision/Test", te_metrics['table_precision'], global_step=epoch)

    writer.add_scalar("Table Recall/Train", tr_metrics['table_recall'], global_step=epoch)
    writer.add_scalar("Table Recall/Test", te_metrics['table_recall'], global_step=epoch)

    writer.add_scalar("Column loss/Train", tr_metrics['column_loss'], global_step=epoch)
    writer.add_scalar("Column loss/Test", te_metrics['column_loss'], global_step=epoch)
    
    writer.add_scalar("Column Acc/Train", tr_metrics['col_acc'], global_step=epoch)
    writer.add_scalar("Column Acc/Test", te_metrics['col_acc'], global_step=epoch)
    
    writer.add_scalar("Column F1/Train", tr_metrics['col_f1'], global_step=epoch)
    writer.add_scalar("Column F1/Test", te_metrics['col_f1'], global_step=epoch)    
    
    writer.add_scalar("Column Precision/Train", tr_metrics['col_precision'], global_step=epoch)
    writer.add_scalar("Column Precision/Test", te_metrics['col_precision'], global_step=epoch)

    writer.add_scalar("Column Recall/Train", tr_metrics['col_recall'], global_step=epoch)
    writer.add_scalar("Column Recall/Test", te_metrics['col_recall'], global_step=epoch)

def display_metrics(epoch, tr_metrics,te_metrics):
    nl = '\n'

    print(f"Epoch: {epoch} {nl}\
            Table Loss -- Train: {tr_metrics['table_loss']:.3f} Test: {te_metrics['table_loss']:.3f}{nl}\
            Table Acc -- Train: {tr_metrics['table_acc']:.3f} Test: {te_metrics['table_acc']:.3f}{nl}\
            Table F1 -- Train: {tr_metrics['table_f1']:.3f} Test: {te_metrics['table_f1']:.3f}{nl}\
            Table Precision -- Train: {tr_metrics['table_precision']:.3f} Test: {te_metrics['table_precision']:.3f}{nl}\
            Table Recall -- Train: {tr_metrics['table_recall']:.3f} Test: {te_metrics['table_recall']:.3f}{nl}\
            {nl}\
            Col Loss -- Train: {tr_metrics['column_loss']:.3f} Test: {te_metrics['column_loss']:.3f}{nl}\
            Col Acc -- Train: {tr_metrics['col_acc']:.3f} Test: {te_metrics['col_acc']:.3f}{nl}\
            Col F1 -- Train: {tr_metrics['col_f1']:.3f} Test: {te_metrics['col_f1']:.3f}{nl}\
            Col Precision -- Train: {tr_metrics['col_precision']:.3f} Test: {te_metrics['col_precision']:.3f}{nl}\
            Col Recall -- Train: {tr_metrics['col_recall']:.3f} Test: {te_metrics['col_recall']:.3f}{nl}\
        ")


def compute_metrics(ground_truth, prediction, threshold = 0.5):
    #https://stackoverflow.com/a/56649983

    ground_truth = ground_truth.int()
    prediction = (torch.sigmoid(prediction) > threshold).int()
    
    TP = torch.sum(prediction[ground_truth==1]==1)
    TN = torch.sum(prediction[ground_truth==0]==0)
    FP = torch.sum(prediction[ground_truth==1]==0)
    FN = torch.sum(prediction[ground_truth==0]==1)

    acc = (TP + TN)/(TP + TN + FP+ FN)
    precision = TP /(FP + TP + 1e-4)
    recall = TP /(FN + TP + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-4)

    metrics = {
        'acc': acc.item(),
        'precision':precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

    return metrics
    
def display(img, table, column, title = 'Original'):
    
    f, ax  = plt.subplots(1,3, figsize = (15,8))
    ax[0].imshow(img)
    ax[0].set_title(f'{title} Image')
    ax[1].imshow(table)
    ax[1].set_title(f'{title} Table Mask')
    ax[2].imshow(column)
    ax[2].set_title(f'{title} Column Mask')
    plt.show()

def get_TableMasks(test_img, model, transform = TRANSFORM, device = config.DEVICE):
    

    image = transform(image = test_img)["image"]
    #get predictions
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        #with torch.cuda.amp.autocast():
        table_out, column_out  = model(image)
        table_out = torch.sigmoid(table_out)
        column_out = torch.sigmoid(column_out)

    #remove gradients

    table_out = (table_out.cpu().detach().numpy().squeeze(0).transpose(1,2,0) > 0.5).astype(int)
    column_out = (column_out.cpu().detach().numpy().squeeze(0).transpose(1,2,0) > 0.5).astype(int)

    return table_out, column_out

def is_contour_bad(c):

    #ref: https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/

	# approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # the contour is 'bad' if it is not a rectangle
    return not len(approx) == 4

def fixMasks(image, table_mask, column_mask):
    
    """
    Fix Table Bounding Box to get better OCR predictions
    """
    table_mask = table_mask.reshape(1024,1024).astype(np.uint8)
    column_mask = column_mask.reshape(1024,1024).astype(np.uint8)
    
    #get contours of the mask to get number of tables
    contours, table_heirarchy = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    table_contours = []
    #ref: https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
    #remove bad contours

    #print(contours)

    for c in contours:
        # if the contour is bad, draw it on the mask


        #if not is_contour_bad(c):
        if cv2.contourArea(c) > 2000:
            table_contours.append(c)
    
    if len(table_contours) == 0:
        return None

    #ref : https://docs.opencv.org/4.5.2/da/d0c/tutorial_bounding_rects_circles.html
    #get bounding box for the contour
    
    table_boundRect = [None]*len(table_contours)
    for i, c in enumerate(table_contours):
        polygon = cv2.approxPolyDP(c, 3, True)
        table_boundRect[i] = cv2.boundingRect(polygon)
    
    #table bounding Box
    table_boundRect.sort()
    
    col_boundRects = []
    for x,y,w,h in table_boundRect:
        
        col_mask_crop = column_mask[y:y+h,x:x+w]
        
        #get contours of the mask to get number of tables
        contours, col_heirarchy = cv2.findContours(col_mask_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #get bounding box for the contour
        boundRect = [None]*len(contours)
        for i, c in enumerate(contours):
            polygon = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(polygon)
            
            #adjusting columns as per table coordinates
            boundRect[i] = (boundRect[i][0] + x ,
                            boundRect[i][1] + y ,
                            boundRect[i][2],
                            boundRect[i][3])
        
        col_boundRects.append(boundRect)
    
    image = image[...,0].reshape(1024, 1024).astype(np.uint8)
    
    #draw bounding boxes
    color = (0,255,0)
    thickness = 4
 
    for x,y,w,h in table_boundRect:
        image = cv2.rectangle(image, (x,y),(x+w,y+h), color, thickness)
    
    return image, table_boundRect, col_boundRects