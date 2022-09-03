# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:20:26 2019

@author: yusuf
"""

#Import libraries
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils
import cv2
import os
import copy

def viewImage(image, name):
    '''Function to display images which are numpy array format'''
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Define PATHs
WORKSPACE_PATH = 'C:/Users/yusuf/Documents/MASAUSTU/YL/MS_THESIS/'
IMG_PATH = 'Mask_seperator/Train2/'
SAVE_PATH = 'C:/Users/yusuf/Documents/MASAUSTU/YL/MS_THESIS/'+IMG_PATH+'Masks'
RAW_IMG_NAME = 'im2.jpg'
MASK_IMG_NAME = 'im2_mask.png'

#Constants
thVal = 30
minDistanceFromCenter = 5

def changeDir(path):
    '''This function prints the current directory and changes the directory to the given path and prints'''
    # Get the current working directory
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    os.chdir(path)

    # Get the current working directory
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

changeDir(WORKSPACE_PATH)

#Read the raw image
img = cv2.imread(IMG_PATH+RAW_IMG_NAME)
# viewImage(img, 'img')

# Read the mask of image
gry = cv2.imread(WORKSPACE_PATH + IMG_PATH + MASK_IMG_NAME, 0)
viewImage(gry,"gry")
# Threshold the mask image as 0s and 1s
ret, th = cv2.threshold(gry, thVal, 1, cv2.THRESH_BINARY)
# viewImage(th*255, "trsh")  # 6


# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(th)
localMax = peak_local_max(D, indices=False, min_distance=minDistanceFromCenter,
                          labels=th)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then apply the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=th)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed algorithm
res = img.copy()
i = 0
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gry.shape, dtype="uint8")
    mask[labels == label] = 255
    # prints each masked segment
    # viewImage(mask, "mask")  # 8
    cv2.imwrite(os.path.join(SAVE_PATH, "mask" + str(i) + ".png"), mask)
    print("mask {} generated..".format(i))
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)

    # res = cv2.bitwise_and(img, img, mask=mask)

    # put a text for each segment---label number
    ((x, y), r) = cv2.minEnclosingCircle(c)
    # cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    i += 1

# viewImage(img, "output")  # 9
# viewImage(res, "res")  # 10
# viewImage(th*255, "trsh")  # 6
