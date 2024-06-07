## test
import re, cv2, os, json
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import segmentation_models as sm
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import decimal
import shutil

##########################################################################################################        
## stage 1
##########################################################################################################  

def skip(line, cell=None):
    '''Skips execution of the current line/cell if line evaluates to True.'''
    if eval(line):
        return

def load_ipython_extension(shell):
    '''Registers the skip magic when the extension loads.'''
    shell.register_magic_function(skip, 'line_cell')

def unload_ipython_extension(shell):
    '''Unregisters the skip magic when the extension unloads.'''
    del shell.magics_manager.magics['cell']['skip']

#Load data and paths
data = pd.read_csv('./data.csv')
images_folder = "./images"
masks_folder = "./masks"
coll_folder = "./collage"

#Obtain a count of images, masks, and observations.
print(f'Total number of images: {len(os.listdir(images_folder))}')
print(f'Total number of image masks: {len(os.listdir(masks_folder))}')
print(f'Length of dataset: {len(data)}')

#Create figure and empty list for axes
axes=[]
fig=plt.figure(figsize=(15, 15))

#Show first 4 images in dataset with corresponding shape.
for a in range(4):
    #Obtain file name and create path.
    file = os.listdir("./images/")[a]
    image_path = os.path.join(images_folder, file) 

    #Read the file image and resize it for show.
    img = cv2.imread(image_path)
    resized_image = cv2.resize(img, (1300, 1500), interpolation = cv2.INTER_AREA)
    
    #Print the resized image and dislpay the shape.
    axes.append(fig.add_subplot(1, 4, a+1) )
    subplot_title=(f"Original Size: {img.shape}")
    axes[-1].set_title(subplot_title)  
    plt.imshow(resized_image)

#Remove ticks from each image.
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

#Plot the image.
fig.tight_layout()    
plt.show()

#Create figure and empty list for axes
axes=[]
fig=plt.figure(figsize=(15, 15))

#Show first 4 images in dataset with corresponding shape.
for a in range(4):
    #Obtain file name and create path.
    file = os.listdir("./masks/")[a]
    image_path = os.path.join(masks_folder, file) 
    
    #Read the file image and resize it for show.
    img = cv2.imread(image_path)
    resized_image = cv2.resize(img, (1300, 1500), interpolation = cv2.INTER_AREA)
    
    #Print the resized image and dislpay the shape.
    axes.append(fig.add_subplot(1, 4, a+1) )
    subplot_title=(f"Original Size: {img.shape}")
    axes[-1].set_title(subplot_title)  
    plt.imshow(resized_image)

#Remove ticks from each image.
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

#Plot the image.
fig.tight_layout()    
plt.show()

#Examine the head of the 'data' DataFrame
pd.set_option('display.max_colwidth', 0)
print(data.head())

#Create function to extract polygon locations from 'location' string.
def location_vals(obvs, x_or_y):
    '''
    Function uses regular expressions to parse the "location" string for each observation.
    Inputs are "obvs" and "x_or_y".
    
    obvs: This simply serves as the string being passed into the function.
    x_or_y: If "x" is entered, then the function extracts all "x" location values. If anything else, then it extracts "y" location values.
    '''
    if x_or_y == 'x':
        x = re.findall(r"\'x\': ([0-9.]*),", obvs)
        return x
    else:
        y = re.findall(r"\'y\': ([0-9.]*)}", obvs)
        return y
    
#Create new column with x and y location values.
data['x_loc_perc'] = data['location'].apply(lambda obvs: location_vals(obvs, 'x'))
data['y_loc_perc'] = data['location'].apply(lambda obvs: location_vals(obvs, 'y'))
print(data.head())

#Creat function to return image size.
def image_size(img_name):
    '''
    The image name from each observation serves as the input.
    The image is then read using cv2, and its shape is returned.
    '''
    image_path = os.path.join(images_folder, img_name)
    img = cv2.imread(image_path)
    return img.shape

#Apply function to each row of DataFrame.
data['shape'] = data['photo_name'].apply(image_size)
print(data.head())

#Save height and weight data as separate features.
data['height'] = data['shape'].apply(lambda x: x[0])
data['width'] = data['shape'].apply(lambda x: x[1])

#Display stats for height and width of images.
print(data[['height', 'width']].describe())

#Make sure that similar all files in each folder have the same location.
for i, j, k in zip(os.listdir(masks_folder), \
                   os.listdir(images_folder), \
                   os.listdir(coll_folder)):
    if (i == j) & (j == k):
        pass
    else:
        print(f'File {i} in one folder does not match name in others.')

##########################################################################################################        
## stage 2
##########################################################################################################  

#Create arrays 
y = np.zeros((1244, 224, 224), dtype='float32')
X = np.zeros((1244, 224, 224, 3), dtype='float32')

for n, image, mask in tqdm(zip(range(1244), os.listdir(images_folder), os.listdir(masks_folder))):
    dir_img = os.path.join(images_folder, image)
    dir_mask = os.path.join(masks_folder, mask)
    
    #Open image, resize it.
    img = cv2.imread(dir_img)
    img = cv2.resize(img, (224, 224))
    #img = ImageOps.exif_transpose(img)
    X[n] = img 
    
    #Open mask image, resize and normalize it.
    msk = cv2.imread(dir_mask)
    msk = cv2.resize(msk, (224, 224))
    
    #Normalize mask values.
    msk = 1.0 * (msk[:, :, 0] > .1)
    
    #Save mask array to y array.
    y[n] = msk   
    
    #Create function to plot images used with segmentation. 
def plot_seg_imgs(array_or_collage, name):
    '''
    This function can be called to print 4 images used with the segmentation model.
    array_or_collage - Accepts any values for arrays, from training arrays to predicted outputs.
        Also accepts 'collage': If this is input, then 4 images from the collages folder will be printed.
    name - What name would you like printed with the number of each image plotted.
    '''
    axes=[]
    fig=plt.figure(figsize=(15, 15))

    for a in range(4):
        
        #Print the resized image and dislpay the shape.
        axes.append(fig.add_subplot(1, 4, a+1))
        subplot_title=(f"{name} Image #{a} Resized")
        axes[-1].set_title(subplot_title)  
        
        if str(array_or_collage) == 'collage':
            img = cv2.imread(os.path.join(coll_folder, os.listdir(coll_folder)[a]))
            img = cv2.resize(img, (224, 224))
            plt.imshow(img)
        else:
            plt.imshow(array_or_collage[a].astype('uint8'))
            
    #Remove ticks from each image.
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    #Plot the image.
    fig.tight_layout()    
    plt.show()
    
#Print first 4 resized meter, resized mask, and collage images.
plot_seg_imgs(X,'Meter Image')
plot_seg_imgs(y, 'Mask')
plot_seg_imgs('collage', 'Collage Image')

#Split data into train and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##########################################################################################################        
## stage 3
##########################################################################################################

