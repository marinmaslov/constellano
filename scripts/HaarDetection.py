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
__version__ = "3.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = """Error! The command should be: py HaarDetection.py --images <images_dir> --masksize <size_in_percentage> --outputname <name> --percision <percision> 
                    --cascade <cascade_dir> --scale <scale> --minNghb <min_neighbour> --json <path_to_json>"""

# Constants --------------------------------------- START
# USED RGB COLORS
RGB_WHITE = (255, 255, 255)
# Constants --------------------------------------- END

def haarDetection(images_dir, output_name, mask_size, percision, output, cascade_path, scale, min_nghb, json_path, cascade_name):
    cascade = cv2.CascadeClassifier(cascade_path)

    # Fetch constellations json
    with open(json_path) as file:
        constellations_data = json.load(file)

    # Should Star mask be applyed?
    constellation_stars_apply = bool(((constellations_data["constellations"])[str(cascade_name)])["draw-star-mask"] == 'true')
    print("[INFO]\tShould constellation masks be applyed: " + str(constellation_stars_apply))

    if constellation_stars_apply:
        # Fetch star data
        constellation_stars_data = (((constellations_data["constellations"])[str(cascade_name)])["star-list"])["stars"]

        # Fetch star connections data
        constellation_connections_data = (((constellations_data["constellations"])[str(cascade_name)])["star-list"])["connections"]

    counter = 0
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            img_rgb, new_file_name, img_rgb_resized = StarDetector.detectStars(images_dir, output_name, mask_size, percision, file, output, counter)
            StarDetector.plotImage(img_rgb)

            gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            constellations = cascade.detectMultiScale(gray, float(scale), int(min_nghb), flags=cv2.CASCADE_SCALE_IMAGE)

            print()
            print("[INFO]\tDetected: " + str(len(constellations)) + " objects that HAAR believes to be '" + str(cascade_name) + "' in file: " + str(file))
            print()

            if len(constellations) == 0:
                print("[ERROR]\tNo objects were detected!")
                continue

            haar_counter = 0
            files_created = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
            for (x, y, w, h) in constellations:
                print("------------------------------------")
                print("Applying star names to detected object at position: [x_start, y_start]: " + str((x, y)) + ", [x_end, y_end]: " + str((x + w, y + h)))
                print("------------------------------------")
                # FETCH THE CROPPED IMAGE OF THE DETECTED CONSTELLATION
                detected_constellation_img = img_rgb_resized[y : y + h, x : x + w]

                #StarDetector.plotImage(detected_constellation_img)


                if constellation_stars_apply:
                    cv2.putText(img_rgb_resized, str(((constellations_data["constellations"])[str(cascade_name)])["name"].title()), (x, y - 20), font, 1.2, RGB_WHITE, 2, cv2.LINE_AA)

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
                        print("------------------------------------")
                        print("Drawing star connections for object at position: [x_start, y_start]: " + str((x, y)) + ", [x_end, y_end]: " + str((x + w, y + h)))
                        print("------------------------------------")
                        for connection in constellation_connections_data:
                            cv2.line(detected_constellation_img, haar_contours_sorted_coordinates[connection[0]], haar_contours_sorted_coordinates[connection[1]], (255, 255, 255), 2)
                            print("[INFO]\tConnecting stars: " + str(haar_contours_sorted_coordinates[connection[0]]) + " and " + str(haar_contours_sorted_coordinates[connection[1]]) + ".")
                        files_created = files_created + 1
                else:
                    cv2.putText(img_rgb_resized, cascade_name, (x, y - 16), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(img_rgb_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

                img_rgb_resized[y : y + h, x : x + w] = detected_constellation_img

            print("[INFO]\tSaving image: " + str(new_file_name))
            cv2.imwrite(new_file_name, img_rgb_resized)

            StarDetector.plotImage(img_rgb_resized)
            
            haar_counter = haar_counter + 1
            counter = counter + 1
            print("------------------------------------")
            print("[INFO]\tTotal files created: " + str(files_created))

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
        opts, args = getopt.getopt(argv, "h", ["images=", "masksize=", "outputname=", "percision=", "cascade=", "scale=", "minNghb=", "json="])
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

    start_time = datetime.now()

    # Call the HAAR Detection function
    output = images_dir + 'output_haar/'

    if not os.path.exists(output) != False:
        print("[INFO]\tCreating directory: " + output)
        os.mkdir(output)

    print("------------------------------------")
    print("Starting HAAR Detector")

    for file in os.listdir(cascade):
        if file.endswith(".xml"):
            cascade_path = str(cascade) + str(file)
            cascade_name = file.split("/")[-1].split(".")[0]
            print("------------------------------------")
            print("Trying to detect object: " + str(cascade_name.upper()))
            print("------------------------------------")

            haarDetection(images_dir, output_name, mask_size, percision, output, cascade_path, scale, min_nghb, json_path, cascade_name)
    
    
    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])