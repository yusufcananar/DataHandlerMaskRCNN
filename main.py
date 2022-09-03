import time

import cv2
import numpy as np
import os
from maskRCNNDataHandler import maskRCNNDataHandler as dataHandler

#Definitions
MAIN_PATH           = 'C:/Users/yusuf/Documents/MASAUSTU/YL/MS_THESIS/Data/'
SUB_IMAGE_NAME      = "sub_image_"
SUB_MASK_NAME       = "sub_mask_"
TRAIN_FOLDER_NAME   = "Train/"

#initialize data handler
dh1 = dataHandler(thVal=1)
dh2 = dataHandler(thVal=30)
images  = []
masks   = []
sub_images  = {}
sub_masks   = {}

#create Train folder
try:
    dh1.mkdir(MAIN_PATH + TRAIN_FOLDER_NAME)
except OSError as e:
    print(e)

# Read images from Data folder
for img in os.listdir(MAIN_PATH):
    # check if the image ends with png
    if (img.endswith(".jpg")):
        print("Reading {} from Data folder.".format(img))
        images.append(img)

# Create sub-images X folder in Data folder and
# Divide image X into sub-images from Data folder and
# write them in sub_image X folder
for img in images:
    name = SUB_IMAGE_NAME + img.split('.')[0]
    targetPath = MAIN_PATH + name
    imPath = MAIN_PATH + img
    dh1.mkdir(targetPath)
    dh1.image_divider(imPath, targetPath=targetPath, sub_im_name="im", imSize = (540, 960), xGridSize = 60, yGridSize = 60)

# Read masks from Data folder
for mask in os.listdir(MAIN_PATH):
    # check if the image ends with png
    if (mask.endswith(".png")):
        print("Reading {} from Data folder.".format(mask))
        masks.append(mask)

# Create sub-masks X folder in Data folder and
# Divide mask X into sub-masks from Data folder and
# write them in sub_mask X folder
for mask in masks:
    name = SUB_MASK_NAME + mask.split('.')[0]
    targetPath = MAIN_PATH + name
    imPath = MAIN_PATH + mask
    dh1.mkdir(targetPath)
    dh1.image_divider(imPath, targetPath=targetPath, sub_im_name="mask", imSize=(540, 960), xGridSize=60, yGridSize=60)

# Read image Y from sub-images X folder
for sub_folder_im in os.listdir(MAIN_PATH):
    # check if the folder starts with sub_image
    if (sub_folder_im.startswith("sub_image")):
        print("Reading {} from Data folder.".format(sub_folder_im))
        sub_images[sub_folder_im] = []
        for img in os.listdir(MAIN_PATH + sub_folder_im):
            # check if the folder ends with .png
            if (img.endswith(".png")):
                print("Reading {} from Data folder.".format(img))
                sub_images[sub_folder_im].append(img)

# Create Train Y folder inside of Train folder and
# Put image Y from sub_image Y folder into Train Y folder
trainYcounter = 0
for listX in sub_images:
    for img in sub_images[listX]:
        imPath = MAIN_PATH + listX + '/' + img
        targetPath = MAIN_PATH + TRAIN_FOLDER_NAME + "Train{}/".format(trainYcounter)
        dh1.mkdir(targetPath)
        trainYcounter += 1
        dh1.copyImage(imPath, targetPath, name='image', extension='.jpg')

# Read mask Y from sub-masks X folder
for sub_folder_mask in os.listdir(MAIN_PATH):
    # check if the folder starts with sub_mask
    if (sub_folder_mask.startswith("sub_mask")):
        print("Reading {} from Data folder.".format(sub_folder_mask))
        sub_masks[sub_folder_mask] = []
        for mask in os.listdir(MAIN_PATH + sub_folder_mask):
            # check if the folder ends with .png
            if (mask.endswith(".png")):
                print("Reading {} from Data folder.".format(mask))
                sub_masks[sub_folder_mask].append(mask)

# Create Masks folder inside of Train Y folder and
# Put mask Y from sub_mask Y folder into Train Y folder
trainYcounter = 0
for idx,listX in enumerate(sub_masks):
    for mask in sub_masks[listX]:
        imPath = MAIN_PATH + listX + '/' + mask
        print(imPath)
        masksFolderPath = MAIN_PATH + TRAIN_FOLDER_NAME + "Train{}/Masks/".format(trainYcounter)
        targetPath = MAIN_PATH + TRAIN_FOLDER_NAME + "Train{}/".format(trainYcounter)
        dh1.mkdir(masksFolderPath)

        dh1.copyImage(imPath, targetPath, name='mask', extension='.png')

        srcPath = MAIN_PATH + TRAIN_FOLDER_NAME + "Train{}/mask.png".format(trainYcounter)
        if idx == 0:
            dh1.generateInstanceMasks(srcPath, masksFolderPath)
        elif idx == 1:
            dh2.generateInstanceMasks(srcPath, masksFolderPath)

        trainYcounter += 1

# Read Train Y -> mask Y

# Run watershed on mask Y and save individual object masks into Train Y -> Masks folder

