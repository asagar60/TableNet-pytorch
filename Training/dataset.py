import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import tqdm
import config
import cv2
import os
import sys

class ImageFolder(nn.Module):
    def __init__(self, df, isTrain = True, transform = None):
        super(ImageFolder, self).__init__()
        self.df = df
        """
        if transform is None:
            if isTrain == True:
                self.transform = A.Compose([
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                        A.Normalize(),
                        ToTensorV2()
                    ])
            else:
                self.test_transform = A.Compose([
                        A.Normalize(),
                        ToTensorV2()
                    ])
        """
        if transform is None:
            self.transform = A.Compose([
                        #A.Resize(width=200, height = 50),
                        #ToTensor --> Normalize(mean, std) 
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value = 255,
                        ),
                        ToTensorV2()
                    ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path, table_mask_path, col_mask_path = self.df.iloc[index, 0], self.df.iloc[index, 1], self.df.iloc[index, 2]
        image = np.array(Image.open("../" + img_path))
        table_image = torch.FloatTensor(np.array(Image.open("../" + table_mask_path))/255.0).reshape(1,1024,1024)
        column_image = torch.FloatTensor(np.array(Image.open("../" + col_mask_path))/255.0).reshape(1,1024,1024)

        """
        augmentations = self.transform(
            image = image, 
            table_mask = table_image,
            column_mask = column_image
        )
        image = augmentations['image']
        table_image = augmentations['table_mask']
        column_image = augmentations['column_mask']
        """
        
        image = self.transform(image = image)['image']


        return {"image":image,"table_image":table_image, "column_image": column_image}


def get_mean_std(train_data, transform):
    dataset = ImageFolder(train_data , transform)

    train_loader = DataLoader(dataset, batch_size=128)

    mean = 0.
    std = 0.
    for img_dict in tqdm.tqdm(train_loader):
        batch_samples = img_dict["image"].size(0) # batch size (the last batch can have smaller size!)
        images = img_dict["image"].view(batch_samples, img_dict["image"].size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)

    print(mean)  #tensor([0.4194, 0.4042, 0.3910])
    print(std) #tensor([0.2439, 0.2402, 0.2372])

#get_mean_std()

if __name__ == '__main__':


    df = pd.read_csv('F:/Deep Learning/appliedai_submissions/Self Case Study/Case Study 2/processed_data_v2.csv')
    dataset = ImageFolder(df[df['hasTable']==1])

    img_num = 0
    for img_dict in dataset:
        save_image(img_dict["image"], f'image_{img_num}.png')
        save_image(img_dict["table_image"], f'table_image_{img_num}.png')
        save_image(img_dict["column_image"], f'column_image_{img_num}.png')

        img_num += 1

        if img_num == 6:
            break

    """
    df = pd.read_csv(config.DATAPATH)
    train_data, test_data  = train_test_split(df, test_size = 0.2, random_state = config.SEED, stratify = df.hasTable)

    dataset = ImageFolder(train_data , isTrain= True)

    train_loader = DataLoader(dataset, batch_size=2)
    for img_dict in train_loader:

        image, table_image, column_image = img_dict['image'], img_dict['table_image'], img_dict['column_image']
        print(image.shape)
        print(table_image.shape)
        print(column_image.shape)
    """