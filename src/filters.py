from src.landmarks_detector import LandmarksDetector

####################################################
# from app [begin]
####################################################

#General libraries
import re, cv2, os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import random
import decimal
import opendatasets as od
from tensorflow import keras
import math
import scipy

from deskew import determine_skew
from typing import Tuple, Union

import segmentation_models as sm
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import warnings
warnings.filterwarnings("ignore")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

####################################################

#image_folder = os.path.join(os.getcwd(), 'images')
#app.config["UPLOAD_FOLDER"] = image_folder

segmentation_model_file = 'final_segmentation_model'
faster_rcnn_path = 'output/model_final.pth' #<-- for cfg.MODEL.WEIGHTS

def prod_resize_input(img_link):
    '''
    Function takes an image and resizes it.
    '''
    img = cv2.imread(img_link)
    img = cv2.resize(img, (224, 224))
    return img.astype('uint8')

#Create function to crop images.
def crop_for_seg(img, bg, mask):
    '''
    Function extracts an image where it overlaps with its binary mask.
    img - Image to be cropped.
    bg - The background on which to cast the image.
    mask - The binary mask generated from the segmentation model.
    '''
    #mask = mask.astype('uint8')
    fg = cv2.bitwise_or(img, img, mask=mask)
    fg_back_inv = cv2.bitwise_or(bg, bg, mask=cv2.bitwise_not(mask))
    New_image = cv2.bitwise_or(fg, fg_back_inv)
    return New_image

def extract_meter(image_to_be_cropped):
    '''
    Function further extracts image such that the meter reading takes up the majority of the image.
    The function finds the edges of the ROI and extracts the portion of the image that contains the entire ROI.
    '''
    where = np.array(np.where(image_to_be_cropped))
    x1, y1, z1 = np.amin(where, axis=1)
    x2, y2, z2 = np.amax(where, axis=1)
    sub_image = image_to_be_cropped.astype('uint8')[x1:x2, y1:y2]
    return sub_image

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    '''
    This function attempts to rotate meter reading images to make them horizontal.
    Its arguments are as follows:

    image - The image to be deskewed (in numpy array format).
    angle - The current angle of the image, found with the determine_skew function of the deskew library.
    background - The pixel values of the boarder, either int (default 0) or a tuple.

    The function returns a numpy array.
    '''
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def resize_aspect_fit(img, final_size: int):
    '''
    Function resizes the image to specified size.

    path - The path to the directory with images.
    final_size - The size you want the final images to be. Should be in int (will be used for w and h).
    write_to - The file you wish to write the images to.
    save - Whether to save the files (True) or return them.
    '''
    im_pil = Image.fromarray(img)
    size = im_pil.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im_pil = im_pil.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im_pil, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    new_im = np.asarray(new_im)
    return np.array(new_im)

def prep_for_ocr(img):
    img = resize_aspect_fit(img, 224)
    output_name = 'test_img.jpg'
    cv2.imwrite(output_name, img)
    return output_name

#Segment input image.
def segment_input_img(img):

    #Resize image.
    img_small = prod_resize_input(img)

    #Open image and get dimensions.
    input_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    input_w = int(input_img.shape[1])
    input_h = int(input_img.shape[0])
    dim = (input_w, input_h)

    #Load model, preprocess input, and obtain prediction.
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    img_small = preprocess_input(img_small)
    img_small = img_small.reshape(-1, 224, 224, 3).astype('uint8')
    model = tf.keras.models.load_model(segmentation_model_file, custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss, 'iou_score' : sm.metrics.iou_score})
    mask = model.predict(img_small)

    #Change type to uint8 and fill in holes.
    mask = mask.astype('uint8')
    mask = scipy.ndimage.morphology.binary_fill_holes(mask[0, :, :, 0]).astype('uint8')

    #Resize mask to equal input image size.
    mask = cv2.resize(mask, dsize=dim, interpolation=cv2.INTER_AREA)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((10,10), np.uint8)

    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    #Create background array.
    bg = np.zeros_like(input_img, 'uint8')

    #Get new cropped image and make RGB.
    New_image = crop_for_seg(input_img, bg, mask)
    New_image = cv2.cvtColor(New_image, cv2.COLOR_BGR2RGB)

    #Extract meter portion.
    extracted = extract_meter(New_image)

    grayscale = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)

    if angle == None:
        angle = 1

    rotated = rotate(extracted, angle, (0, 0, 0))
    return rotated

def get_reading(image_path):
    '''
    This is the main function for the pipeline.
    It takes an input image path as its only argument.
    It then carries out all the necessary steps to extract a meter reading.
    The output is the reading.

    NOTE: Due to having to load and generate predictions from two models,
    this script may take a while to run.
    '''

    #Segment image.
    segmented = segment_input_img(image_path)

    #Prep image and save path.
    prepped_path = prep_for_ocr(segmented)

    #Class labels.
    labels = ['number', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #List for storing meter readings.
    list_of_img_reading = []

    #Configure model parameters.
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = 'output/model_final.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    predictor = DefaultPredictor(cfg)

    #Read prepped image and obtain prediction.
    im = cv2.imread(prepped_path)
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
    pred_class_names = list(map(lambda x: labels[x], final_predictions))

    #Add decimal point to list of digits depending on number of bounding boxes.
    if len(pred_class_names) == 5:
        pass
    else:
        pred_class_names.insert(5, '.')

    #Combine digits and convert them into a float.
    combine_for_float = "".join(pred_class_names)
    meter_reading = float(combine_for_float)

    return meter_reading

####################################################
# from app [end]
####################################################

def dodge(x, y):
    return cv2.divide(x, 255 - y, scale=256)


def burn(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)


def image2pencilSketch(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_inv = 255 - image_gray
    image_gray_inv_blur = cv2.GaussianBlur(image_gray_inv, (21, 21), sigmaX=0, sigmaY=0)
    image_dodged = dodge(image_gray, image_gray_inv_blur)
    image_result = burn(image_dodged, image_gray_inv_blur)
    return image_result


def image2gray(image):
    image_result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_result


def landmarks2image(image, background, landmarks):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [landmarks], -1, (255, 255, 255), -1)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = cv2.multiply(cv2.subtract(mask, 50), 2)

    r_min = np.min(landmarks[:, 1])
    r_max = np.max(landmarks[:, 1])
    c_min = np.min(landmarks[:, 0])
    c_max = np.max(landmarks[:, 0])
    r_center = (r_max + r_min) // 2
    c_center = (c_max + c_min) // 2

    image_lips = image[r_min:r_max, c_min:c_max]
    mask_lips = mask[r_min:r_max, c_min:c_max]

    lips = cv2.bitwise_and(image_lips, image_lips, mask=mask_lips)
    lips = cv2.resize(lips, (0, 0), fx=2, fy=2)
    mask_lips = cv2.resize(mask_lips, (0, 0), fx=2, fy=2)

    lips_w, lips_h, _ = lips.shape
    lips_x_min = r_center - lips_w // 2
    lips_y_min = c_center - lips_h // 2

    mask_new = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_new[lips_x_min:lips_x_min + lips_w, lips_y_min:lips_y_min + lips_h] = mask_lips

    lips_new = np.zeros(image.shape, dtype=np.uint8)
    lips_new[lips_x_min:lips_x_min + lips_w, lips_y_min:lips_y_min + lips_h] = lips
    foreground = lips_new.astype(float)  # Convert uint8 to float

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = mask_new.astype(float) / 255
    alpha = cv2.merge((alpha, alpha, alpha))

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    result = cv2.add(foreground, background)

    return result


def face_eyes_lips(image_path):
    image = cv2.imread(image_path)
    background = image.astype(float)
    landmarks_detector = LandmarksDetector()
    all_face_landmarks = landmarks_detector.get_landmarks(image)
    lips_landmarks_indexes = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]
    left_eye_landmarks_indexes = [35, 41, 40, 42, 39, 37, 33, 36]
    right_eye_landmarks_indexes = [89, 95, 94, 96, 93, 91, 87, 90]

    for face_landmarks in all_face_landmarks:
        left_eye_landmarks = np.array([face_landmarks[i] for i in left_eye_landmarks_indexes])
        background = landmarks2image(image, background, left_eye_landmarks)

        right_eye_landmarks = np.array([face_landmarks[i] for i in right_eye_landmarks_indexes])
        background = landmarks2image(image, background, right_eye_landmarks)

        lips_landmarks = np.array([face_landmarks[i] for i in lips_landmarks_indexes])
        background = landmarks2image(image, background, lips_landmarks)

    result = background.astype(np.uint8)
    return result


def image2cartoon(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_blur = cv2.medianBlur(image_gray, 5)
    edges = cv2.adaptiveThreshold(image_gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    image_color = cv2.bilateralFilter(image, 9, 250, 250)
    image_result = cv2.bitwise_and(image_color, image_color, mask=edges)
    return image_result


def who_knows_me_best(image, face_parts_detector):
    embedding, aligned_image, left_eye_image, right_eye_image, nose_image, lips_image = face_parts_detector(image)

    # cv2.imwrite('output/aligned_image.jpg', aligned_image)
    # cv2.imwrite('output/left_eye_image.jpg', left_eye_image)
    # cv2.imwrite('output/right_eye_image.jpg', right_eye_image)
    # cv2.imwrite('output/nose_image.jpg', nose_image)
    # cv2.imwrite('output/lips_image.jpg', lips_image)

    face_dataset = np.load('face_dataset.npy', allow_pickle=True)
    face_embedding = [face['embedding'] for face in face_dataset]
    distances = np.linalg.norm(face_embedding - embedding, axis=1)
    nearest_faces_indices = distances.argsort()[:3]

    for i in nearest_faces_indices:
        face = face_dataset[i]
        image_path = face['image_path']
        image = cv2.imread(image_path)
        embedding, aligned_image, left_eye_image, right_eye_image, nose_image, lips_image = face_parts_detector(image)

        # cv2.imwrite(f'output/aligned_image_{i}.jpg', aligned_image)
        # cv2.imwrite(f'output/left_eye_image_{i}.jpg', left_eye_image)
        # cv2.imwrite(f'output/right_eye_image_{i}.jpg', right_eye_image)
        # cv2.imwrite(f'output/nose_image_{i}.jpg', nose_image)
        # cv2.imwrite(f'output/lips_image_{i}.jpg', lips_image)


if __name__ == "__main__":
    image_path = "input/photos/file_0.jpg"
    # result = image2cartoon(image_path)
    # result = image2gray(image_path)
    # result = image2pencilSketch(image_path)
    result = face_eyes_lips(image_path)
    cv2.imshow('output', result)
    cv2.waitKey(0)
