#############################
import cv2, os, re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import decimal
import shutil
import albumentations as A
import segmentation_models as sm
import keras.backend as K
from keras.layers import Flatten
#############################
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
#import torch, torchvision
import os, sys
#############################

sm.set_framework('tf.keras')
sm.framework()

BASE_DIR = ''
IMG_DIR = BASE_DIR + "images/"
MASK_DIR = BASE_DIR + "masks/"
COLL_DIR = BASE_DIR + "collage/"
OCR_CROP = BASE_DIR + "ocr_crop/"
ROTATED_DIR = BASE_DIR + "rotated_imgs/"

data = pd.read_csv('data.csv')

imgs = os.listdir(IMG_DIR)
masks = os.listdir(MASK_DIR)

print(f"Img files: {len(imgs)}. ---> {imgs[:3]}")
print(f"Mask files :  {len(masks)}. ---> {masks[:3]}")
print(f'Length of dataset: {len(data)}')

rows = 1
columns = 2

for img, mask in zip(imgs[:5], masks[:5]):
    img_path = os.path.join(IMG_DIR, img)
    mask_path = os.path.join(MASK_DIR, mask)
    
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    
    fig.add_subplot(rows, columns, 2)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    plt.imshow(mask, interpolation=None)
    plt.axis('off')
    plt.title("Ground truth")

#plt.show()

print(data.head())

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
    
data['x_loc_perc'] = data['location'].apply(lambda obvs: location_vals(obvs, 'x'))
data['y_loc_perc'] = data['location'].apply(lambda obvs: location_vals(obvs, 'y'))
print(data.head())

def image_size(img_name):
    image_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(image_path)
    return img.shape

#Apply function to each row of DataFrame.
data['shape'] = data['photo_name'].apply(image_size)
print(data.head())

data.to_csv('new_data.csv')

#Save height and weight data as separate features.
data['height'] = data['shape'].apply(lambda x: x[0])
data['width'] = data['shape'].apply(lambda x: x[1])

#Display stats for height and width of images.
data[['height', 'width']].describe()

out_rgb = []
out_mask = []

counter_empty = 0

for p_img, p_mask in zip(imgs, masks):   
    img_path = os.path.join(IMG_DIR, p_img)
    mask_path = os.path.join(MASK_DIR, p_mask)
    
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256)) / 255.
    
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (256, 256))
    mask = 1.0 * (mask[:, :, 0] > .1)    
    if 1 not in mask: counter_empty+=1
    mask = np.expand_dims(mask, axis=-1)

    out_rgb += [img]
    out_mask += [mask]

out_rgb = np.array(out_rgb, dtype = 'float32')
out_mask = np.array(out_mask, dtype = 'float32')

print(f"Mask files: {out_mask.shape[0]}. ---> Empty: {counter_empty}")

print(out_rgb.shape, out_mask.shape)

rows = 1
columns = 2

for img, mask in zip(out_rgb[:5], out_mask[:5]):
    #img_path = os.path.join(IMG_DIR, image)
    #mask_path = os.path.join(MASK_DIR, msk)
    
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    #img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
    plt.imshow(img)
    plt.axis('off')
    plt.title("Resized Image")
    
    fig.add_subplot(rows, columns, 2)
    #mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    plt.imshow(mask, interpolation=None)
    plt.axis('off')
    plt.title("Resized Ground truth")
    
#plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    out_rgb, 
    out_mask, 
    test_size=0.1, 
    shuffle=True)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

aug = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(50, 101), height=256, width=256, p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
    ],p=1),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.8)])

def make_image_gen(X_train, y_train, aug, batch_size):
    aug_x = []
    aug_y = []
    while True:
        for i in range(X_train.shape[0]): 
            augmented = aug(image=X_train[i], mask=y_train[i])
            x, y = augmented['image'],  augmented['mask']
            aug_x.append(x)
            aug_y.append(y)
            if len(aug_x)>=batch_size:
                yield np.array(aug_x, dtype = 'float32'), np.array(aug_y, dtype = 'float32')
                aug_x, aug_y=[], []
                
train_gen = make_image_gen(X_train, y_train, aug, 32)
aug_x, aug_y = next(train_gen)
print(np.shape(aug_x), np.shape(aug_y))

def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
    dice = tf.keras.backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=15),
    tf.keras.callbacks.ModelCheckpoint(filepath='./water_meters.keras', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=5, 
                                   verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)   
]

def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = tf.keras.layers.Flatten()(inputs)
    targets = tf.keras.layers.Flatten()(targets)
    
    intersection = tf.keras.backend.sum(tf.keras.backend.dot(targets, inputs))
    dice = (2*intersection + smooth) / (tf.keras.backend.sum(targets) + tf.keras.backend.sum(inputs) + smooth)
    return 1 - dice


ALPHA = 0.8
GAMMA = 2

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
    inputs = tf.keras.layers.Flatten()(inputs)
    targets = tf.keras.layers.Flatten()(targets)
    
    BCE = tf.keras.losses.binary_crossentropy(targets, inputs)
    BCE_EXP = tf.keras.backend.exp(-BCE)
    focal_loss = tf.keras.backend.mean(alpha * tf.math.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss


model = sm.Unet('efficientnetb0', classes=1, input_shape=(256, 256, 3), activation='sigmoid', encoder_weights='imagenet')
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=FocalLoss, metrics = [dice_coef] )
generator = make_image_gen(X_train, y_train, aug, 16)
#model.fit(generator, steps_per_epoch = 200, epochs=50, callbacks = callbacks,validation_data = (X_test, y_test))


preds = model.predict(X_test)

rows = 1
columns = 3

for img, pred, mask in zip(X_test[:5], preds[:5], y_test[:5]):
    
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis('off')
    plt.title("Prediction")
    
    fig.add_subplot(rows, columns, 3)
    plt.imshow(mask, interpolation=None)
    plt.axis('off')
    plt.title("Ground truth")
    
#Create function to crop images.
def crop(img, bg, mask) -> np.array:
    '''
    Function takes image, background, and mask, and crops the image.
    The cropped image should correspond only with the positive portion of the mask.
    '''
    fg = cv2.bitwise_or(img, img, mask=mask) 
    fg_back_inv = cv2.bitwise_or(bg, bg, mask=cv2.bitwise_not(mask))
    New_image = cv2.bitwise_or(fg, fg_back_inv)
    return New_image

ocr_path = 'ocr_crop'

for n, image, mask in zip(range(len(os.listdir(IMG_DIR))), os.listdir(IMG_DIR), os.listdir(MASK_DIR)):
    dir_img = os.path.join(IMG_DIR, image)
    dir_mask = os.path.join(MASK_DIR, mask)
    
    #Read images and masks.
    img = cv2.imread(dir_img).astype('uint8')
    mask = cv2.imread(dir_mask).astype('uint8')
    
    #Get dimensions of image.
    h, w, _ = img.shape
    
    #Ensure mask is binary, and create black background in shape of image.
    mask = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), (w, h)) # Resize image
    bg = np.zeros_like(img, 'uint8') # Black background

    #Crop image based on mask and make it RBG.
    New_image = crop(img,bg,mask)
    New_image = cv2.cvtColor(New_image, cv2.COLOR_BGR2RGB)

    #Extract portion of image where meter reading is.
    #Use min and max x and y coordinates to obtain final image.
    where = np.array(np.where(New_image))
    x1, y1, z1 = np.amin(where, axis=1)
    x2, y2, z2 = np.amax(where, axis=1)
    sub_image = New_image.astype('uint8')[x1:x2, y1:y2]

    #Write image to file
    cv2.imwrite(os.path.join(ocr_path , image), sub_image)

for img in imgs:
    img_path = os.path.join(OCR_CROP, img)
    rotated_img_path = os.path.join(ROTATED_DIR, img)#.split('.jpg')[0]+'.png')
    image = io.imread(img_path)
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, resize=True) * 255
    io.imsave(rotated_img_path, rotated.astype(np.uint8))
    
rotated = os.listdir(ROTATED_DIR)
print(f"Rotated files :  {len(rotated)}. ---> {rotated[:3]}")


def resize_aspect_fit(path, final_size: int, write_to, save=True):
    '''
    Function resizes the image to specified size.
    
    path - The path to the directory with images.
    final_size - The size you want the final images to be. Should be in int (will be used for w and h).
    write_to - The file you wish to write the images to. 
    save - Whether to save the files (True) or return them.
    '''   
    for item in os.listdir(path):
        im = Image.open(path+item)
        f, e = os.path.splitext(path+item)
        size = im.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        im = im.resize(new_image_size, Image.Resampling.LANCZOS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        if save==True:
            cv2.imwrite(os.path.join(resize_for_rcnn, item), np.array(new_im))
        else:
            return np.array(new_im)
        
#Reshape all images to 224x224x3 size, while retaining aspect. 
#IMPORTANT FOR PREPROCESSING IMAGES

if os.path.exists('./resized_for_rcnn') == False:
    os.mkdir('resized_for_rcnn')
else:
    pass

#Specify argument values for resize function.
resize_for_rcnn = './resized_for_rcnn'
path = './ocr_crop/'
final_size = 224

resize_aspect_fit(path, final_size, resize_for_rcnn)
