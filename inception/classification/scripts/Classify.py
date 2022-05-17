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

from datetime import datetime

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

# Constants --- START
COMMAND_FORMAT = "Error! The command should be: py Detect.py --model <model_dir> --images <images_dir> --output <output_dir_name>"

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
# Constants --- END

def loadImage(image_file):
    pil_image = Image.open(image_file)
    pil_image = ImageOps.fit(pil_image, (299, 299), Image.Resampling.LANCZOS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(image_file, format="JPEG", quality=100)

    img = tf.io.read_file(image_file)
    img = tf.image.decode_jpeg(img, channels=3)

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

    detector = hub.load(model_name).signatures['serving_default']

    #print(detector.signatures)

    img = loadImage(images_dir)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key:value.numpy() for key,value in result.items()}

    results = np.squeeze(result["dense"])
    labels = list(loadLabels(label_file).values())

    
    results_dict = {}
    for i in range(len(results)):
        results_dict[results[i]] = labels[i]
    results_dict = collections.OrderedDict(sorted(results_dict.items(), reverse=True))


    print("[INFO]\tResults:")
    for key in results_dict:
        format_key = "{:.2f}".format(key)
        if key < 0:
            print("Score = " + str(format_key) + "%\tfor: " + str(results_dict[key]))
        else:
            print("Score =  " + str(format_key) + "%\tfor: " + str(results_dict[key]))

    print("------------------------------------")
    print("[INFO]\tObject is probably: " + str(results_dict[next(iter(results_dict))]))
    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])