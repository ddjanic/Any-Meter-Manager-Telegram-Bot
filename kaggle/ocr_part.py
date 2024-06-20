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
#os.mkdir('content')
#os.chdir('content')

#Register directories for training, testing, and validation datasets.
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "train/_annotations.coco.json", "./content/train")
register_coco_instances("my_dataset_val", {}, "valid/_annotations.coco.json", "./content/valid")
register_coco_instances("my_dataset_test", {}, "test/_annotations.coco.json", "./content/test")

#Visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

axes=[]
fig=plt.figure(figsize=(10, 10))

#View three sample training images.
for i, d in enumerate(random.sample(dataset_dicts, 3)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
        
    #Print the resized image and dislpay the shape.
    axes.append(fig.add_subplot(1, 3, i+1))
    plt.imshow(vis.get_image()[:, :, ::-1])

#Remove ticks from each image.
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

#Plot the image.
fig.tight_layout()    
plt.show()

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

#Create trainer. 
class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

#Configure model parameters.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  #Initialize training from model zoo.
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #Adjusted up as mAP was still increasing after 1500.
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11 #classes + 1 | We are trying to detect 10 different digits (0-9). 
cfg.TEST.EVAL_PERIOD = 500

#Need to clear GPU memory, as Kaggle's limits sometimes cause memory error.
from numba import cuda
cuda.select_device(0)
cuda.close()

#Train model.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()