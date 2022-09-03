# DataHandlerMaskRCNN
Image manipulations for MaskRCNN

## About
maskRCNNDataHandler.py is a library created to manipulate images and masks to build a better Mask RCNN dataset structure.         
It is capable of;   
1. dividing images into sub-images,    
2. extracting all instances from a single mask using watershed algorithm,    
3. viewing images,    
4. saving images,    
5. copying images,    
6. making directories.   


### Data Handler Resulting File Structure
![Data Handler Resulting File Structure](resources/dataHandlerStructure(1).png)

### How it works?   
Running main.py results as in the figure above. However, before running;   
1. Data folder must be exists,   
2. imX.jpg and imX_mask.png files must be in the Data folder.   

Everything else is generated according to imX.jpg and imX_mask.png files.  
