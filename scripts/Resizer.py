#!/usr/bin/env python
""" Constellano Star Recognition
Python script for resizing images.

Command format:
    py Resizer.py --images <images_dir> --size <final_image_size> --grayscale <0_if_images_sould_be_bw>
"""

import sys
import os
import cv2
import getopt
import numpy as np
from datetime import datetime

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.0.2"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py Resizer.py --images <images_dir> --size <final_image_size> --grayscale <0_if_images_sould_be_bw>"


def resize_image(img, newRows, newCols):
    return cv2.resize(img, (int(newCols), int(newRows)), interpolation=cv2.INTER_AREA)


def fill(img, size):
    filled = np.zeros((int(size), int(size), 3), np.uint8)

    rows, cols, _ = img.shape
    filled_rows, filled_cols, _ = filled.shape

    if rows > cols:
        free_spaces = int(size) - cols
        for i in range(0, filled_rows - 1):
            for j in range(0, filled_cols - 1):
                if j >= int(free_spaces / 2) and j <= int(free_spaces / 2 + cols - 1):
                    filled[i, j] = img[i, int(j - free_spaces / 2)]
                else:
                    filled[i, j] = 0

    if cols > rows:
        free_spaces = int(size) - rows
        for i in range(0, filled_rows - 1):
            for j in range(0, filled_cols - 1):
                if i >= int(free_spaces / 2) and i <= int(free_spaces / 2 + rows - 1):
                    filled[i, j] = img[int(i - free_spaces / 2), j]
                else:
                    filled[i, j] = 0
    return filled

def resize(img, size):
    rows, cols, _ = img.shape
    resized = 0
    filled = 0

    if rows > cols:
        rowsRatio = float(int(size) / rows)
        newCols = int(rowsRatio * cols)
        resized = resize_image(img, int(size), newCols)
        if not cols == rows:
            filled = fill(resized, int(size))
    else:
        colsRatio = float(int(size) / cols)
        newRows = int(colsRatio * rows)
        resized = resize_image(img, newRows, int(size))
        if not cols == rows:
            filled = fill(resized, int(size))

    return filled, resized


def main(argv):
    images_dir = ''
    image_size = ''
    grayscale = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "size=", "grayscale="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--size"):
            image_size = arg
        elif opt in ("--grayscale"):
            grayscale = arg


    location = str(images_dir)
    output = location + 'resized/'

    if not os.path.exists(output) != False:
        print("Creating directory: " + output)
        os.mkdir(output)

    start_time = datetime.now()
    counter = 0
    for file in os.listdir(location):
        if file.endswith(".jpg"):
            # PREPARE OUTPUT NAME
            zeros = "00000"
            zeros_counter = len(str(counter))
            while zeros_counter > 0:
                zeros = zeros[:-1]
                zeros_counter = zeros_counter - 1

            new_file_name = str(output + "resized_" + str(zeros) + str(counter) + ".jpg")
            print("[INFO]\tResizing file: " + str(file) + " (saving resized image to: " + str(new_file_name) + ").")

            # READ IMAGE (RGB)
            img = cv2.imread(location + file)

            filled, resized = resize(img, image_size)

            rows, cols, _ = img.shape
            if not cols == rows:
                if int(grayscale) == 0:
                    img_bw = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(new_file_name, img_bw)
                else:
                    cv2.imwrite(new_file_name, filled)
            else:
                if int(grayscale) == 0:
                    img_bw = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(new_file_name, img_bw)
                else:
                    cv2.imwrite(new_file_name, resized)

            counter = counter + 1

    print("------------------------------------")
    print("Total number of resized images: " + str(counter))
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])