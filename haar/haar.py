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
    image = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["cascade=", "image="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--cascade"):
            cascade = arg
        elif opt in ("--image"):
            image = arg

    # HAAR --------------------------------------- START
    cascade = cv2.CascadeClassifier(cascade)

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    constellations = cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in constellations:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)
    # HAAR --------------------------------------- END

if __name__ == "__main__":
    main(sys.argv[1:])
