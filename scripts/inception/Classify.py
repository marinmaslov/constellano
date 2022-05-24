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
import inspect

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


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import Resizer


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
COMMAND_FORMAT = "Error! The command should be: py Classify.py --model <model_dir> --images <images_dir> --output <output_dir_name>"

DO_FINE_TUNING = False
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
# Constants --- END

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
    labels = loadLabels(label_file)
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape = IMAGE_SIZE + (3,)),
        hub.KerasLayer(str(model_name), trainable = DO_FINE_TUNING),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(len(labels), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + IMAGE_SIZE+(3,))

    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            img = tf.keras.utils.load_img(
                images_dir + file, target_size=(299, 299)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            print("[INFO]\tDetected " + str(labels[np.argmax(score)].upper()) + " (with accuracy: " + str(100 * np.max(score)) + "%) in image: " + str(file))
    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])