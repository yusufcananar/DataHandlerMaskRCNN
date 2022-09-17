#Import libraries
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import cv2
import os


class maskRCNNDataHandler():
    def __init__(self, thVal=1, minDistanceFromCenter=4):
        # Constants
        self.thVal = thVal
        self.minDistanceFromCenter = minDistanceFromCenter

    def viewImage(self, name, src):
        '''Function to display images which are numpy array format'''
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getMaskImage(self, path):
        gry = cv2.imread(path, 0)
        # Threshold the mask image as 0s and 1s
        ret, th = cv2.threshold(gry, self.thVal, 1, cv2.THRESH_BINARY)
        return th

    def watershed(self, path):

        th = self.getMaskImage(path)
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(th)

        localMax = peak_local_max(D, indices=False, min_distance=self.minDistanceFromCenter,
                                  labels=th)
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then apply the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=th)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        return labels



    def imageSaver(self, src, targetPath, name='example', extension='.png'):
        cv2.imwrite(os.path.join(targetPath, name + extension), src)

    def generateInstanceMasks(self, imPath, targetPath):
        # loop over the unique labels returned by the Watershed algorithm
        i = 0
        labels = self.watershed(imPath)
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background' so simply ignore it
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw it on the mask
            mask = np.zeros(self.getMaskImage(imPath).shape, dtype="uint8")
            mask[labels == label] = 255

            self.imageSaver(mask, targetPath=targetPath, name='Mask{}'.format(i))
            print("mask {} generated and saved to {}".format(i, targetPath))
            i += 1

    def image_divider(self, imPath, targetPath, sub_im_name='im', imSize = (540, 960), xGridSize = 60, yGridSize = 60):

        im = cv2.imread(imPath)
        # Split the image into grids and save them
        counter = 0
        for j in range(0, imSize[0], yGridSize):
            for i in range(0, imSize[1], xGridSize):
                roi = im[j:j + yGridSize, i:i + xGridSize]
                self.imageSaver(roi, targetPath, name='{}{}'.format(sub_im_name, counter))
                print("Sub-image {} generated and saved to {}".format(counter, targetPath))
                counter += 1

        print("Total # of sub-images : ", counter+1)

    def mkdir(self, newPath):
        if not os.path.exists(newPath):
            os.makedirs(newPath)
            print("Directory '% s' created" % newPath)
        else:
            print("Directory '% s' is already exists" % newPath)

    def copyImage(self, srcPath, targetPath, name='image', extension='.png'):
        src = cv2.imread(srcPath)
        print(srcPath)
        self.imageSaver(src, targetPath=targetPath, name=name, extension=extension)
