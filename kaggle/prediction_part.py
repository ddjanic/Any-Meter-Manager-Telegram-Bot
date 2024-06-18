#General libraries
import re, cv2, os, json, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random
import decimal
import shutil
import opendatasets as od
import keras
import math
import scipy

#Image deskew libraries.
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from typing import Tuple, Union

import segmentation_models as sm

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

import warnings
warnings.filterwarnings("ignore")

import keras.backend as K

def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
    dice = tf.keras.backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

ALPHA = 0.8
GAMMA = 2

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
    inputs = tf.keras.layers.Flatten()(inputs)
    targets = tf.keras.layers.Flatten()(targets)
    
    BCE = tf.keras.losses.binary_crossentropy(targets, inputs)
    BCE_EXP = tf.keras.backend.exp(-BCE)
    focal_loss = tf.keras.backend.mean(alpha * tf.math.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

## Segmentation part

img = './Water counter.jpg'

#Specify files for model weights.
segmentation_model_file = './water_meters.keras'
faster_rcnn_path = './model_final.pth' 

img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256)) / 255.
plt.imshow(img)
img_to_pred=np.expand_dims(img,axis=0)
plt.show()

model = tf.keras.models.load_model(segmentation_model_file, custom_objects={'FocalLoss': FocalLoss, 'dice_coef' : dice_coef})
pred = model.predict(img_to_pred)
pred_to_show = pred.squeeze()
plt.imshow(pred_to_show)
plt.show()