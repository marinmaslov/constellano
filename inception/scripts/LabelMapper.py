#!/usr/bin/env python
""" Constellano Label Mapper
Python script for creation of label maps (.pbtxt).

Command format:
    py LabelMapper.py --images <images_dir> --output <output_name>
"""

import sys
import os
import cv2
import getopt
import numpy as np
from datetime import datetime

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.1.2"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py LabelMapper.py --images <images_dir> --output <output_name>"

def convertClassesToLabelMap(classes, start=1):
    msg = ''
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + "\tid: " + str(id) + "\n"
        msg = msg + "\tname: '" + name + "'\n}\n\n"
    return msg[:-1]

def main(argv):
    images_dir = ''
    output = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "output="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--output"):
            output = arg

    start_time = datetime.now()

    label_names = os.listdir(images_dir)

    # Save the label map
    print("[INFO]\tCreating label_map.pbtxt in " + str(images_dir))
    label_file_path = output + "label_map.pbtxt"
    with open(label_file_path, 'w') as f:
        f.write(convertClassesToLabelMap(label_names))

    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])