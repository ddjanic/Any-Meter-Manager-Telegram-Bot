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
sm.set_framework('tf.keras')
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
#import torch, torchvision

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
