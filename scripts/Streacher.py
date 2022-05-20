#!/usr/bin/env python
""" Constellano Image Rotator
Python script for streaching images.

Command format:
    py Streacher.py --images <images_dir>
"""

import sys
import os
import cv2
import math
import getopt
import numpy as np
from datetime import datetime

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "1.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = """Error! The command should be: py Streacher.py --images <images_dir>"""

def streachImage(img, size):
    streached = np.zeros((int(size), int(size), 3), np.uint8)
    for i in range(0, size - 1):
        for j in range(0, size - 1):
            streached[i, j] = 0

    rows, cols, _ = img.shape
    rows_offset = int((size - rows) / 2)
    cols_offset = int((size - cols) / 2)

    streached[rows_offset : rows_offset + rows, cols_offset : cols_offset + cols] = img[0 : rows, 0 : cols]
    return streached

def main(argv):
    images_dir = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images_dir = arg

    location = str(images_dir)
    output = location + 'streached/'

    if not os.path.exists(output) != False:
        print("Creating directory: " + output)
        os.mkdir(output)

    start_time = datetime.now()
    counter = 0
    for file in os.listdir(location):
        if file.endswith(".jpg"):
            print("[INFO]\tStreaching image: " + str(file))

            # PREPARE OUTPUT NAME
            zeros = "00000"
            zeros_counter = len(str(counter))
            while zeros_counter > 0:
                zeros = zeros[:-1]
                zeros_counter = zeros_counter - 1
                new_file_name = str(output + "streached_" + str(zeros) + str(counter) + ".jpg")

            # READ IMAGE (RGB)
            img = cv2.imread(location + file)

            cols, rows, _ = img.shape

            new_dimension = int(math.ceil(math.sqrt(cols * cols + rows * rows)))

            if new_dimension % 2 != 0:
                new_dimension = new_dimension + 1

            # Streach image
            streached = streachImage(img, new_dimension)
            cv2.imwrite(new_file_name, streached)

            counter = counter + 1

    print("------------------------------------")
    print("Total number of rotated images: " + str(counter)),
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])