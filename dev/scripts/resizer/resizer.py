#!/usr/bin/env python
""" Constellano Star Recognition
Python script for finding biggest and brightest stars in images and overlaying a target over them.

Command format:
    py resizer.py -d <image_dir> -s <wanted_image_size>

Command example:
    py resizer.py -d img/ -s 500
"""

import sys
import os
import cv2
import numpy as np
import getopt

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.0.0"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py resizer.py --images <images_dir> --size <final_image_size> --grayscale <0_if_images_sould_be_bw> --log <wanted_log_level>"


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


def main(argv):
    images_dir = ''
    image_size = ''
    grayscale = ''
    log_level = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "size=", "grayscale=", "log="])
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
        elif opt in ("--log"):
            log_level = arg

    # Algorithm --------------------------------------- START
    location = str(images_dir)
    output = location + 'resized/'

    if not os.path.exists(output) != False:
        print("Creating directory: " + output)
        os.mkdir(output)

    counter = 0

    for file in os.listdir(location):
        if file.endswith(".png"):
            # PREPARE OUTPUT NAME
            zeros = "00000"
            zeros_counter = len(str(counter))
            while zeros_counter > 0:
                zeros = zeros - "0";
                zeros_counter = zeros_counter - 1

            new_file_name = str(output + "resized_" + str(zeros) + str(counter) + ".png")

            print("\033[2;32;40m[INFO]\033[0;0m" + "\tResizing file: " + str(file) + " (saving resized image to: " + str(new_file_name) + ")")
            # READ IMAGE (RGB)
            img = cv2.imread(location + file)

            rows, cols, _ = img.shape
            resized = 0
            filled = 0

            if rows > cols:
                rowsRatio = float(int(image_size) / rows)
                newCols = int(rowsRatio * cols)
                resized = resize_image(img, int(image_size), newCols)
                if not cols == rows:
                    filled = fill(resized, int(image_size))
            else:
                colsRatio = float(int(image_size) / cols)
                newRows = int(colsRatio * rows)
                resized = resize_image(img, newRows, int(image_size))
                if not cols == rows:
                    filled = fill(resized, int(image_size))

            if not cols == rows:
                if grayscale == 0:
                    img_bw = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(new_file_name, img_bw)
                else:
                    cv2.imwrite(new_file_name, filled)
            else:
                if grayscale == 0:
                    img_bw = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(new_file_name, img_bw)
                else:
                    cv2.imwrite(new_file_name, resized)
            cv2.waitKey(0)
            counter = counter + 1

    print("------------------------------------")
    print("Total number of resized images: " + str(counter)),
    # Algorithm --------------------------------------- END


if __name__ == "__main__":
    main(sys.argv[1:])
