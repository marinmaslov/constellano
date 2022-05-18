#!/usr/bin/env python
""" Constellano .XML to .RECORD converter
Python script for .xml to .record (tensorflow) conversation.

Command format:
    py ConvertXmlToRecord.py --images <images_dir> --labels <path_to_label_map> --output <output_dir_name>
"""

import sys
import os
import cv2
import getopt
import numpy as np
from datetime import datetime


import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
tf = tf.compat.v1
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile
from collections import namedtuple

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.1.2"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py ConvertXmlToRecord.py --images <images_dir> --labels <path_to_label_map> --output <output_dir_name>"

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (filename,
                    width,
                    height,
                    member.find('name').text,
                    int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text),
                    int(bndbox.find('ymax').text),
                    )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def class_text_to_int(row_label, label_map_dict):
    return label_map_dict[row_label]

def create_tf_example(group, path, label_map_dict):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map_dict))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(argv):
    images = ''
    labels = ''
    output = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "labels=", "output="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images = arg
        elif opt in ("--labels"):
            labels = arg
        elif opt in ("--output"):
            if arg[-1] == "/":
                output = arg
            else:
                output = arg + "/"

    start_time = datetime.now()


    if not os.path.exists(output) != False:
        print("Creating directory: " + output)
        os.mkdir(output)

    #label_map = label_map_util.load_labelmap(labels)
    #label_map_dict = label_map_util.get_label_map_dict(label_map)

    label_map_dict = label_map_util.get_label_map_dict(labels)

    print(output)
    writer = tf.python_io.TFRecordWriter(output + 'train' + ".record")
    examples = xml_to_csv(images)
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, images, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()

    print('Successfully created the TFRecord file: {}'.format(output))
    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")

if __name__ == "__main__":
    main(sys.argv[1:])