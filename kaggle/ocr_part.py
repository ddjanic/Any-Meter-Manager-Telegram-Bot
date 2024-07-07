import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

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
import gc

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
cfg.SOLVER.MAX_ITER = 1500 #Adjusted up as mAP was still increasing after 1500.
cfg.SOLVER.STEPS = ()
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11 #classes + 1 | We are trying to detect 10 different digits (0-9). 
cfg.TEST.EVAL_PERIOD = 500
#cfg.MODEL.DEVICE = 'cpu'

#Need to clear GPU memory, as Kaggle's limits sometimes cause memory error.
from numba import cuda
cuda.select_device(0)
cuda.close()

gc.collect()
torch.cuda.empty_cache()
#os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

# Usage example
min_memory_available = 2 * 1024 * 1024 * 1024  # 2GB
clear_gpu_memory()
wait_until_enough_gpu_memory(min_memory_available)

#Train model.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=True)
#trainer.train()

#%%skip True

#Get prediction metrics for test dataset.
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(trainer.model, val_loader, evaluator)

###########################

import re
import pandas as pd
#Obtain model and parameters for obtaining predictions from test images.
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") <-- Changed path to saved file. 
cfg.MODEL.WEIGHTS = './output/model_final.pth'#'../../input/water-meter-ocr-images/output/model_final.pth'
cfg.DATASETS.TEST = ("my_dataset_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_test")
#metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) <-- Can't be used when changed saved dir.
metadata = MetadataCatalog.get("my_dataset_train")
class_catalog = metadata.thing_classes
#Import necessary libraries.
from detectron2.utils.visualizer import ColorMode
import glob

#List for storing meter readings.
list_of_img_reading = []

#Obtain test predictions.
#for i, imageName in enumerate(glob.glob('test/*jpg')): <-- Once reading from saved file, have to change.
for i, imageName in enumerate(os.listdir('./content/test')): #('../../input/water-meter-ocr-images/content/test')):
    #Read image and get output information.
    if 'coco' in imageName:
        pass
    else:
        #im = cv2.imread(os.path.join('../../input/water-meter-ocr-images/content/test', imageName))
        im = cv2.imread(os.path.join('./content/test', imageName))
        outputs = predictor(im)
    
        #Find predicted boxes and labels.
        instances = outputs['instances']
        coordinates = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        pred_classes = outputs['instances'].pred_classes.cpu().tolist()
    
        #Obtain list of all predictions and the leftmost x-coordinate for bounding box.
        pred_list = []
        for pred, coord in zip(pred_classes, coordinates):
            pred_list.append((pred, coord[0]))
    
        #Sort the list based on x-coordinate in order to get proper order or meter reading.
        pred_list = sorted(pred_list, key=lambda x: x[1])
    
        #Get final order of identified classes, and map them to class value.
        final_predictions = [x[0] for x in pred_list]
        pred_class_names = list(map(lambda x: class_catalog[x], final_predictions))
    
        #Add decimal point to list of digits depending on number of bounding boxes.
        if len(pred_class_names) == 5:
            pass
        else:
            pred_class_names.insert(5, '.')
    
        #Combine digits and convert them into a float.
        combine_for_float = "".join(pred_class_names)
        meter_reading = float(combine_for_float)
    
        #Visualize prediction.  
        metadata_model = MetadataCatalog.get("mydataset")
        v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.8)
    
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        parsed_name = re.findall(r"([a-zA-Z0-9]*_?[a-zA-Z0-9]*_?[a-zA-Z0-9]*_?[a-zA-Z0-9]*_?[a-zA-Z0-9]*)", imageName)
        list_of_img_reading.append((parsed_name[0], meter_reading))
    
        #Plot only subset of images.
        if i % 5 == 0:
            plt.figure()
            plt.imshow(out.get_image()[:, :, ::-1])
            plt.xticks([])
            plt.yticks([])
        else:
            pass

data = pd.read_csv('./content/new_data.csv')

#Create DataFrame from predictions, and format photo name to the same style as in provided DataFrame.
df_predicted = pd.DataFrame(list_of_img_reading, columns=['photo_name', 'reading'])
df_predicted.photo_name = df_predicted['photo_name'].apply(lambda x: x + '.jpg')

#Upload provided data to DataFrame.
df_provided = data.copy()

#Create list to get info from both DataFrames where photo names overlap.
provided_and_predicted = []
pred_list = list(df_predicted['photo_name'].values)

for image_name, predicted_value in df_predicted.values:
    provided_row = df_provided[df_provided['photo_name'] == image_name]
    provided_value = float(provided_row['value'].values)
    provided_and_predicted.append((image_name, provided_value, predicted_value))
    
compiled_df = pd.DataFrame(provided_and_predicted, columns=['image_name', 'ground_truth', 'predicted_value'])
print(compiled_df.head())

#Round each value to 3 decimal places and calculate difference between predicted value and ground truth.
compiled_df['ground_truth'] = compiled_df['ground_truth'].apply(lambda x: round(x, 3))
compiled_df['predicted_value'] = compiled_df['predicted_value'].apply(lambda x: round(x, 3))
compiled_df['difference'] = abs((compiled_df['ground_truth']) - (compiled_df['predicted_value']))

#Calculate percentage difference between ground truth and predicted value.
compiled_df['percent_diff'] = (compiled_df['difference'] / ((compiled_df['ground_truth'] + compiled_df['predicted_value']) / 2) * 100)
print(compiled_df.head())

#Round each value to 3 decimal places and calculate difference between predicted value and ground truth.
compiled_df['ground_truth'] = compiled_df['ground_truth'].apply(lambda x: round(x, 3))
compiled_df['predicted_value'] = compiled_df['predicted_value'].apply(lambda x: round(x, 3))
compiled_df['difference'] = abs((compiled_df['ground_truth']) - (compiled_df['predicted_value']))

#Calculate percentage difference between ground truth and predicted value.
compiled_df['percent_diff'] = (compiled_df['difference'] / ((compiled_df['ground_truth'] + compiled_df['predicted_value']) / 2) * 100)
print(compiled_df.head())

#Obtain list of percentages over 5% error and with no error.
over_5_error = [x for x in compiled_df['percent_diff'] if x >= .05]
no_error = [x for x in compiled_df['percent_diff'] if x == 0]

print(f'Total number of accurate predictions: {len(no_error)}. Percentage of total: {round(len(no_error) / len(compiled_df), 3)}')
print(f'Total number of predictions with less than 5% error: {len(compiled_df) - len(over_5_error)}. Percentage of Total: {round((len(compiled_df) - len(over_5_error)) / len(compiled_df), 3)}')
print(f'Total number of predictions with over 5% error: {len(over_5_error)}. Percentage of total: {round(len(over_5_error) / len(compiled_df), 3)}')
     