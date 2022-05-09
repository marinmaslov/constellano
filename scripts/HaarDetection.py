#!/usr/bin/env python
""" Constellano Star HAAR Recognition
Python script for detecting stars using HAAR cascade.

Command format:
    py HaarDetection.py --images <images_dir> --masksize <size_in_percentage> --outputname <name> --percision <percision> --cascade <cascade_file> --scale <scale> --minNghb <min_neighbour>
"""

import sys
import os
import cv2
import getopt
import json
from datetime import datetime

import StarDetector

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.0.0"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = """Error! The command should be: py HaarDetection.py --images <images_dir> --masksize <size_in_percentage> --outputname <name> --percision <percision> 
                    --cascade <cascade_file> --scale <scale> --minNghb <min_neighbour> --json <path_to_json>"""

# Constants --------------------------------------- START
# USED RGB COLORS
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RBG_GREEN = (0, 255, 0)
# Constants --------------------------------------- END

def haarDetection(images_dir, output_name, mask_size, percision, output, cascade, scale, min_nghb, json_path):
    start_time = datetime.now()

    cascade_name = cascade.split("/")[-1].split(".")[0]
    print("[INFO]\tTrying to detect object: " + str(cascade_name))
    cascade = cv2.CascadeClassifier(cascade)

    # Fetch constellations json
    with open(json_path) as file:
        constellations_data = json.load(file)

    # Fetch star data
    constellation_stars_data = (((constellations_data["constellations"])[str(cascade_name)])["star-list"])["stars"]

    # Fetch star connections data
    constellation_connections_data = (((constellations_data["constellations"])[str(cascade_name)])["star-list"])["connections"]

    counter = 0
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            img_rgb, new_file_name, img_rgb_resized = StarDetector.detectStars(images_dir, output_name, mask_size, percision, file, output, counter)

            gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            constellations = cascade.detectMultiScale(gray, float(scale), int(min_nghb), flags=cv2.CASCADE_SCALE_IMAGE)

            print("[INFO]\tDetected: " + str(len(constellations)) + " objects that HAAR believes to be '" + str(cascade_name) + "' in file: " + str(file))

            if len(constellations) == 0:
                break

            haar_counter = 0
            files_created = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
            for (x, y, w, h) in constellations:
                # FETCH THE CROPPED IMAGE OF THE DETECTED CONSTELLATION
                detected_constellation_img = img_rgb_resized[y : y + h, x : x + w]

                StarDetector.plotImage(detected_constellation_img)

                cv2.putText(img_rgb_resized, str(((constellations_data["constellations"])[str(cascade_name)])["name"].title()), (x, y - 20), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

                # GET CONTOURS ON CROPPED DETECTED IMAGE
                haar_img_bw = cv2.cvtColor(detected_constellation_img, cv2.COLOR_BGR2GRAY)
                haar_thresholds = cv2.threshold(haar_img_bw, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                haar_contours = cv2.findContours(haar_thresholds, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

                # Create sorted list of dict values
                haar_contours_sorted = []
                for contour in haar_contours:
                    haar_contours_sorted.append(cv2.contourArea(contour))
                haar_contours_sorted.sort(reverse = True)

                haar_contours_sorted_coordinates = []
                star = 0
                for area in haar_contours_sorted[:len(constellation_stars_data)]:
                    for contour in haar_contours:
                        if area == cv2.contourArea(contour):
                            x1, y1, w1, h1 = cv2.boundingRect(contour)
                            haar_contours_sorted_coordinates.append((int(x1 + (w1/2)), int(y1 + (h1/2))))
                            cv2.putText(detected_constellation_img, str(((constellation_stars_data[str(star)])["name"].title())[:3]), (int(x1 - 45), int(y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, float(0.7), (255, 255, 255), 2)
                            print("[INFO]\tApplying name for star: " + str((constellation_stars_data[str(star)])["name"].title()) + ", at position:" + str((int(x1 + (w1/2)), int(y1 + (h1/2)))))
                    star = star + 1

                # DRAW LINES
                if len(constellation_connections_data) != len(haar_contours_sorted_coordinates):
                    print("[ERROR]\tIncompatible object detected!")
                else:
                    for connection in constellation_connections_data:
                        cv2.line(detected_constellation_img, haar_contours_sorted_coordinates[connection[0]], haar_contours_sorted_coordinates[connection[1]], (255, 255, 255), 2)
                        print("[INFO]\tConnecting stars: " + str(haar_contours_sorted_coordinates[connection[0]]) + " and " + str(haar_contours_sorted_coordinates[connection[1]]) + ".")
                    files_created = files_created + 1

                img_rgb_resized[y : y + h, x : x + w] = detected_constellation_img

            print("[INFO]\tSaving image: " + str(new_file_name))
            cv2.imwrite(new_file_name, img_rgb_resized)
            
            haar_counter = haar_counter + 1
            counter = counter + 1
            print("------------------------------------")
            print("[INFO]\tTotal files created: " + str(files_created))

    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


def main(argv):
    images_dir = ''
    mask_size = ''
    output_name = ''
    percision = ''
    cascade = ''
    scale = ''
    min_nghb = ''
    json_path = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "masksize=", "outputname=", "percision=", "log=", "cascade=", "scale=", "minNghb=", "json="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--masksize"):
            mask_size = float(arg)
        elif opt in ("--outputname"):
            output_name = arg
        elif opt in ("--percision"):
            percision = float(arg)
        elif opt in ("--cascade"):
            cascade = arg
        elif opt in ("--scale"):
            scale = arg
        elif opt in ("--minNghb"):
            min_nghb = arg
        elif opt in ("--json"):
            json_path = arg

    # Call the HAAR Detection function
    output = images_dir + 'output_haar/'

    if not os.path.exists(output) != False:
        print("[INFO]\tCreating directory: " + output)
        os.mkdir(output)

    haarDetection(images_dir, output_name, mask_size, percision, output, cascade, scale, min_nghb, json_path)

if __name__ == "__main__":
    main(sys.argv[1:])