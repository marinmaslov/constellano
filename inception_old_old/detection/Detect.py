#!/usr/bin/env python
""" Constellano Tensorflow2 Image Classification Script
Python script for image classification using Tensorflow2 and Keras.

Command format:
    py Classify.py --model <model_dir> --images <images_dir> --output <output_dir_name>
"""

import os
import sys
import cv2
import getopt

import collections
import math

import matplotlib.pylab as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import tensorflow as tf
import tensorflow_hub as hub






import torch
from torchvision import transforms


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util














from datetime import datetime


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile



__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

# Constants --- START
COMMAND_FORMAT = "Error! The command should be: py Detect.py --model <model_dir> --images <images_dir> --output <output_dir_name>"

IMAGE_SIZE = (297, 297)
BATCH_SIZE = 32
# Constants --- END

def run_inference_for_single_image(model, image):
    #print(image)
    #image = cv2.imread(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (int(299), int(299)), interpolation=cv2.INTER_AREA)
    """
    #image = np.asarray(image)
    im_arr32 = image.astype(np.float32)
    im_tensor = torch.tensor(im_arr32)
    #im_tensor = im_tensor.unsqueeze(0)

    im_tensor = tf.expand_dims(im_tensor, 0)
    """

    # Convert image to float32
    converted_img  = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(converted_img)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    print("INPUT TENSOR")
    print(input_tensor.shape)

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

def show_inference(model, image_path, category_index):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #image_np = np.array(Image.open(image_path))
    # Actual detection.
    image_np = loadImage(image_path)
    #image_np = cv2.resize(image_np, (int(879), int(879)), interpolation=cv2.INTER_AREA)
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    display(Image.fromarray(image_np))
















def loadImage(image_file):
    pil_image = Image.open(image_file)
    pil_image = ImageOps.fit(pil_image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    pil_image_bw = pil_image.convert('L')
    pil_image_bw.save(image_file, format="JPEG", quality=100)

    img = tf.io.read_file(image_file)
    img = tf.image.decode_jpeg(img, channels=0)

    return img

def loadLabels(label_file):
    item_id = None
    item_name = None
    items = {}
    
    with open(label_file, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items

# Main function
def main(argv):
    model_name = ''
    images_dir = ''
    label_file = ''
    output = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["model=", "images=", "labels=", "output="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--model"):
            model_name = arg
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--labels"):
            label_file = arg
        elif opt in ("--output"):
            if arg[-1] == "/":
                output = arg
            else:
                output = arg + "/"

    start_time = datetime.now()

    if not os.path.exists(output) != False:
        print("[INFO]\tCreating directory: " + output)
        os.mkdir(output)

    print("[INFO]\tBuilding model with", model_name)

    # LOAD THE MODEL
    model = tf.saved_model.load(str(model_name))
    

    print(model)

    # LOAD THE LABELS
    category_index = label_map_util.create_category_index_from_labelmap(label_file, use_display_name=True)

    print(category_index)

    print(model.signatures['serving_default'].output_dtypes)
    print(model.signatures['serving_default'].output_shapes)

    print(model.signatures['serving_default'].inputs)

    #show_inference(model, images_dir, category_index)

    
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            show_inference(model, images_dir + file, category_index)
    
    
    """

    img = loadImage(images_dir)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    results = model(converted_img)

    
    #result = {key:value.numpy() for key,value in results.items()}

    print(results)

    
    label_id_offset = 0
    image_np_with_detections = converted_img.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
    keypoints = result['detection_keypoints'][0]
    keypoint_scores = result['detection_keypoint_scores'][0]

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

    plt.figure(figsize=(24,32))
    plt.imshow(image_np_with_detections[0])
    plt.show()
    """

    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])