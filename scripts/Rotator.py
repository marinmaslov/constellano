#!/usr/bin/env python
""" Constellano Image Rotator
Python script for Rotating images.

Command format:
    py Rotator.py --images <images_dir> --maxangle <max_roration_angle> --anglestep <angle_rotation_step>
"""

import sys
import os
import cv2
import getopt
from datetime import datetime

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "1.0.0"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = """Error! The command should be: py Rotator.py --images <images_dir> --maxangle <max_roration_angle>
                                        --anglestep <angle_rotation_step>"""

def rotateImage(img, i):
    # FIND CENTER OF IMAGE
    h, w, _ = img.shape
    (cX, cY) = (w // 2, h // 2)

    # ROTATE IMAGE BY i DEGREES
    print("[INFO]\tRotating image by: " + str(i) + " degrees!")
    M = cv2.getRotationMatrix2D((cX, cY), float(i), 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def main(argv):
    images_dir = ''
    max_angle = ''
    angle_step = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "maxangle=", "anglestep="])
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
        elif opt in ("--anglestep"):
            angle_step = float(arg)

    location = str(images_dir)
    output = location + 'rotated/'

    if not os.path.exists(output) != False:
        print("Creating directory: " + output)
        os.mkdir(output)

    start_time = datetime.now()
    counter = 0
    for file in os.listdir(location):
        if file.endswith(".jpg"):
            print("[INFO]\tRotating image: " + str(file))

            i = angle_step
            while i <= int(max_angle):
                # PREPARE OUTPUT NAME
                zeros = "00000"
                zeros_counter = len(str(counter))
                while zeros_counter > 0:
                    zeros = zeros[:-1]
                    zeros_counter = zeros_counter - 1
                    new_file_name = str(output + "rotated_" + str(zeros) + str(counter) + ".jpg")

                print("[INFO]\tSaving rotated (angle: " + str(float(i)) + ") image to: " + str(new_file_name))
                
                # READ IMAGE (RGB)
                img = cv2.imread(location + file)
                rotated = rotateImage(img, i)

                cv2.imwrite(new_file_name, rotated)

                i = i + angle_step
                counter = counter + 1

    print("------------------------------------")
    print("Total number of rotated images: " + str(counter)),
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])