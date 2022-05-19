#!/usr/bin/env python
""" Constellano Renamer Script
Python script for image renaming.

Command format:
    py Renamer.py --images <images_dir> --output <output_name>
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

COMMAND_FORMAT = "Error! The command should be: py Renamer.py --images <images_dir> --output <output_name>"

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


    location = str(images_dir)

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

            new_file_name = str(location + output + "_" + str(zeros) + str(counter) + ".jpg")
            print("[INFO]\tRenaming file: " + str(file) + " to " + output + "_" + str(zeros) + str(counter) + ".jpg" + " (saving renamed image to: " + str(new_file_name) + ").")

            # READ IMAGE (RGB)
            img = cv2.imread(location + file)

            # REMOVE FILE
            os.remove(location + file)

            # SAVE RENAMED IMAGE (RGB)
            cv2.imwrite(new_file_name, img)

            counter = counter + 1

    print("------------------------------------")
    print("Total number of renamed images: " + str(counter))
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])