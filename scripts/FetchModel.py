#!/usr/bin/env python

""" Constellano Model Fetcher
Python script for fetching of tensorflow models.

Command format:
    py Resizer.py --images <images_dir> --size <final_image_size> --grayscale <0_if_images_sould_be_bw>
"""

import sys
import getopt
import urllib.request
import tarfile

from datetime import datetime

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.1.2"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py FetchModel.py --model <model_name> --tf <tensorflow_version>"

MODEL_MAP_TF1 = {
    "mask_rcnn_inception_v2_coco": "http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz",
    "faster_rcnn_inception_resnet_v2_atrous_coco": "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz",
    "faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco": "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz",
    "faster_rcnn_inception_v2_coco": "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz",
    "ssd_inception_v2_coco": "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz",
    "ssd_mobilenet_v2_coco": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
    "ssd_mobilenet_v1_coco": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz",
    "ssd_mobilenet_v3_small_coco": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz",
    "ssd_mobilenet_v3_large_coco": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz"
}

MODEL_MAP_TF2 = {
    "ssd_mobilenet_v2_320x320_coco17": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
    "faster_rcnn_resnet50_v1_640x640_coco17": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz",
    "faster_rcnn_inception_resnet_v2_640x640_coco17": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz",
    "mask_rcnn_inception_resnet_v2_1024x1024_coco17": "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
}

def main(argv):
    model = ''
    tf = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["model=", "tf="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--model"):
            model = arg
        elif opt in ("--tf"):
            tf = arg

    start_time = datetime.now()

    if int(tf) == 1:
        print("[INFO]\tFetching model: " + str(model) + " from: " + str(MODEL_MAP_TF1.get(model)))
        thetarfile = MODEL_MAP_TF1.get(model)
        ftpstream = urllib.request.urlopen(thetarfile)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        print("[INFO]\tExtracting model...")
        thetarfile.extractall()

    if int(tf) == 2:
        print("[INFO]\tFetching model: " + str(model) + " from: " + str(MODEL_MAP_TF2.get(model)))
        thetarfile = MODEL_MAP_TF2.get(model)
        ftpstream = urllib.request.urlopen(thetarfile)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        print("[INFO]\tExtracting model...")
        thetarfile.extractall()

    if int(tf) != 1 or int(tf) != 2:
        print("[ERROR]\tTensorflow version can only be 1 or 2!")

    print("------------------------------------")
    print("[INFO]\tModel extracted!")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")

if __name__ == "__main__":
    main(sys.argv[1:])