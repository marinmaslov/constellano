#!/usr/bin/env python
""" Constellano Image Rotator
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
import time

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py resizer.py --images <images_dir> --maxangle <max_roration_angle> --log <wanted_log_level>"


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
    max_angle = ''
    log_level = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "maxangle=", "log="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--maxangle"):
            max_angle = float(arg)
        elif opt in ("--log"):
            log_level = arg

    # Algorithm --------------------------------------- START
    location = str(images_dir)
    output = location + 'rotated/'

    if not os.path.exists(output) != False:
        print("Creating directory: " + output)
        os.mkdir(output)

    counter = 0

    for file in os.listdir(location):
        if file.endswith(".png"):
            print("\033[2;32;40m[INFO]\033[0;0m" + "\tRotating file:\t" + str(file))
            i = 5
            while i <= int(max_angle):
                # PREPARE OUTPUT NAME
                zeros = "00000"
                zeros_counter = len(str(counter))
                while zeros_counter > 0:
                    zeros = zeros[:-1]
                    zeros_counter = zeros_counter - 1

                new_file_name = str(output + "rotated_" + str(zeros) + str(counter) + ".png")

                print("\033[2;32;40m[INFO]\033[0;0m" + "\tSaving rotated (angle: " + str(float(i)) + ") image to:\t" + str(new_file_name))
                # READ IMAGE (RGB)
                img = cv2.imread(location + file)

                # FIND CENTER OF IMAGE
                h, w, _ = img.shape
                (cX, cY) = (w // 2, h // 2)

                # ROTATE IMAGE BY i DEGREES
                M = cv2.getRotationMatrix2D((cX, cY), float(i), 1.0)
                rotated = cv2.warpAffine(img, M, (w, h))
                cv2.imwrite(new_file_name, rotated)
                cv2.waitKey(0)
                i = i + 5
                counter = counter + 1

    print("------------------------------------")
    print("Total number of rotated images: " + str(counter)),
    # Algorithm --------------------------------------- END


if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    print("\033[2;32;40m[INFO]\033[0;0m" + "\tTotal execution time: " + str((time.time() - start_time)) + " seconds.\033[0;0m")
