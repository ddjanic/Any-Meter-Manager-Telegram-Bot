import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

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


#Import Relevant Libraries
import random
from PIL import Image, ImageOps
from tqdm import tqdm
import decimal
import shutil
import opendatasets as od
import os
import matplotlib.pyplot as plt
import cv2, re
import pandas as pd

#Create new directory and add model images to it.
os.mkdir('content')
os.chdir('content')