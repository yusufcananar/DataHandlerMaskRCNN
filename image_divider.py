import cv2
import numpy as np
import os

def viewImage(image, name):
    '''Function to display images which are numpy array format'''
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imSize = (540, 960)
gridSquare = 16*9
xGridSize = 60
yGridSize = 60

# WORKSPACE_PATH = 'C:\Users\yusuf\Documents\MASAUSTU\YL\MS_THESIS\Train'
path = 'C:/Users/yusuf/Documents/MASAUSTU/YL/MS_THESIS/Mask_seperator/Train2/'
im_name = 'im2.jpg'

im = cv2.imread(path + im_name, 0)
print(np.shape(im))
viewImage(im, "raw_im")


# Split the image into grids and save them
counter = 0
for j in range(0,imSize[0],yGridSize):
    for i in range(0,imSize[1], xGridSize):
        roi = im[j:j + yGridSize, i:i + xGridSize]
        print('i : {} , j : {}'.format(i,j))
        print(counter)
        viewImage(roi, "roi")
        counter += 1



