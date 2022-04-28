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
__version__ = "1.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py resizer.py -d <images_dir> -s <image_size>"

def main(argv):
    cascade = ''
    images = ''
    scale = ''
    min_nghb = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["cascade=", "images=", "scale=", "minNghb="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--cascade"):
            cascade = arg
        elif opt in ("--images"):
            images = arg
        elif opt in ("--scale"):
            scale = arg
        elif opt in ("--minNghb"):
            min_nghb = arg

    # HAAR --------------------------------------- START
    counter = 0
    output = images + "haar_detection/"

    if not os.path.exists(output) != False:
        print("\033[2;32;40m[INFO]\033[0;0m" + "\tCreating directory: " + output)
        os.mkdir(output)

    cascade = cv2.CascadeClassifier(cascade)

    for file in os.listdir(images):
        if file.endswith(".jpg"):
            zeros = "00000"
            zeros_counter = len(str(counter))
            while zeros_counter > 0:
                zeros = zeros[:-1]
                zeros_counter = zeros_counter - 1

            new_file_name = str(output + "haar_detected_" + str(zeros) + str(counter) + ".jpg")

            img = cv2.imread(images + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            constellations = cascade.detectMultiScale(gray, float(scale), int(min_nghb), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x,y,w,h) in constellations:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            print("\033[2;32;40m[INFO]\033[0;0m" + "\tSaving image: " + str(new_file_name))
            cv2.imwrite(new_file_name, img)
            counter = counter + 1
    print("------------------------------------")
    print("\033[2;32;40m[INFO]\033[0;0m" + "\t\033[2;44;47mTotal files created:\t" + str(counter) + "\033[0;0m")
    # HAAR --------------------------------------- END

if __name__ == "__main__":
    main(sys.argv[1:])
