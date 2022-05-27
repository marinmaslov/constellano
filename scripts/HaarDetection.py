#!/usr/bin/env python
""" Constellano Star HAAR Recognition
Python script for detecting stars using HAAR cascade.

Command format:
    py HaarDetection.py --images <images_dir> --masksizeMin <size_in_percentage>  --masksizeMax <size_in_percentage> --outputname <name> --percision <percision> --cascade <cascade_file> --scale <scale> --minNghb <min_neighbour>
"""

import sys
import os
import cv2
import getopt
import math
import json
from datetime import datetime

import StarDetector
import Streacher

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "5.2.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = """Error! The command should be: py HaarDetection.py --images <images_dir> --masksizeMin <size_in_percentage> --masksizeMax <size_in_percentage> --outputname <name>
                                                    --percisionMin <percision> --percisionMax <percision> --cascade <cascade_dir> --scale <scale> --minNghb <min_neighbour> --json <path_to_json> --plot <not_0_for_plotting_of_images --streach <0_not>"""

# Constants --------------------------------------- START
# USED RGB COLORS
RGB_WHITE = (255, 255, 255)
# Constants --------------------------------------- END

def haarDetection(images_dir, output_name, mask_size_min, mask_size_max, percision_min, percision_max, output, cascade_path, scale, min_nghb, json_path, cascade_name, plot_images, streach):
    cascade = cv2.CascadeClassifier(cascade_path)

    # Fetch constellations json
    with open(json_path) as file:
        constellations_data = json.load(file)

    # Should Star mask be applyed?
    constellation_stars_apply = bool(((constellations_data["constellations"])[str(cascade_name)])["draw-star-mask"] == 'true')
    print("[INFO]\tShould constellation masks be applyed: " + str(constellation_stars_apply))

    # Fetch star data
    constellation_stars_data = (((constellations_data["constellations"])[str(cascade_name)])["star-list"])["stars"]

    # Fetch star connections data
    constellation_connections_data = (((constellations_data["constellations"])[str(cascade_name)])["star-list"])["connections"]

    counter = 0
    detected_constellations = 0
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            detected_constellations = 0
            percision = percision_max
            while percision >= percision_min:
                mask_size = mask_size_max
                while mask_size >= mask_size_min:
                    print("------------------------------------")
                    print("[INFO]\tDetecting stars with percision: " + str(percision))
                    print("[INFO]\tDetecting stars with mask size percentage: " + str(mask_size))
                    print("------------------------------------")

                    if streach:
                        image = cv2.imread(images_dir + file)
                        cols, rows, _ = image.shape
                        new_dimension = int(math.ceil(math.sqrt(cols * cols + rows * rows)))

                        if new_dimension % 2 != 0:
                            new_dimension = new_dimension + 1
                        image = Streacher.streachImage(image, new_dimension)
                        img_rgb, new_file_name, img_rgb_resized, _ = StarDetector.detectStars(images_dir, output_name, mask_size, percision, file, output, counter, None, None, image)
                    else:
                        img_rgb, new_file_name, img_rgb_resized, _ = StarDetector.detectStars(images_dir, output_name, mask_size, percision, file, output, counter, None, None, None)
                    
                    if plot_images != 0:
                        StarDetector.plotImage(img_rgb)

                    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

                    dimensions = gray.shape
                    if dimensions[0] > dimensions[1]:
                        max_detection = dimensions[0]
                    else:
                        max_detection = dimensions[1]
                    
                    constellations, rejectLevels, levelWeights = cascade.detectMultiScale3(gray, float(scale), int(min_nghb), minSize = (100, 100), maxSize = (max_detection, max_detection), flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels = True) #minSize = (200, 200), maxSize = (3400, 3400), 
                    
                    print("[INFO]\tHAAR levels: ", rejectLevels, levelWeights)

                    print()
                    print("[INFO]\tDetected: " + str(len(constellations)) + " objects that HAAR believes to be '" + str(cascade_name) + "' in file: " + str(file))
                    print()

                    mask_size = round(mask_size - 0.01, 2)
                    if len(constellations) == 0:
                        print("[ERROR]\tNo objects were detected!")
                        continue

                    haar_counter = 0
                    files_created = 0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    for (x, y, w, h) in constellations:
                        if detected_constellations != 0:
                            print("[WARN]\tObject already found! Breaking out of haar detection loop!")
                            break

                        print("------------------------------------")
                        print("Applying star names to detected object at position: [x_start, y_start]: " + str((x, y)) + ", [x_end, y_end]: " + str((x + w, y + h)))
                        print("------------------------------------")
                        # FETCH THE CROPPED IMAGE OF THE DETECTED CONSTELLATION
                        detected_constellation_img = img_rgb_resized[y : y + h, x : x + w]

                        if plot_images != 0:
                            StarDetector.plotImage(detected_constellation_img)

                        # Check if detection contains more stars then needed
                        try:
                            preview, _, _, star_count = StarDetector.detectStars(images_dir, output_name, 0.06, percision, file, output, counter, None, None, detected_constellation_img)

                            if plot_images != 0:
                                StarDetector.plotImage(preview)

                            if  len(constellation_stars_data) > int(star_count) or int(math.ceil(len(constellation_stars_data) * 0.20) + len(constellation_stars_data)) > int(star_count):
                                print("[ERROR]\tIncompatible object detected! Discarding.")

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
                                            cv2.putText(detected_constellation_img, str(((constellation_stars_data[str(star)])["name"].title())[:3]), (int(x1 - 45), int(y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, float(0.7), RGB_WHITE, 2)
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
                                        cv2.line(detected_constellation_img, haar_contours_sorted_coordinates[connection[0]], haar_contours_sorted_coordinates[connection[1]], RGB_WHITE, 2)
                                        print("[INFO]\tConnecting stars: " + str(haar_contours_sorted_coordinates[connection[0]]) + " and " + str(haar_contours_sorted_coordinates[connection[1]]) + ".")
                                    files_created = files_created + 1
                            else:
                                constellation_name = cascade_name.replace("_", " ")
                                
                                if y - 40 < 0:
                                    cv2.putText(img_rgb_resized, constellation_name.title(), (x, y + h + 100), font, 2.5, RGB_WHITE, 10, cv2.LINE_AA)
                                else:
                                    cv2.putText(img_rgb_resized, constellation_name.title(), (x, y - 40), font, 2.5, RGB_WHITE, 10, cv2.LINE_AA)
                                cv2.rectangle(img_rgb_resized, (x, y), (x + w, y + h), (115, 200, 255), 8)

                            img_rgb_resized[y : y + h, x : x + w] = detected_constellation_img

                            print("[INFO]\tSaving image: " + str(new_file_name))
                            cv2.imwrite(new_file_name, img_rgb_resized)

                            if plot_images != 0:
                                StarDetector.plotImage(img_rgb_resized)

                            haar_counter = haar_counter + 1
                            counter = counter + 1
                            detected_constellations = detected_constellations + 1
                        except IndexError:
                            print("[ERROR]\tArea has no stars! Discarding!")

                    print("------------------------------------")
                    print("[INFO]\tTotal files created: " + str(files_created))

                    if detected_constellations != 0:
                        print("------------------------------------")
                        print("[INFO]\tDetected constellation is: " + constellation_name.title())
                        print("[INFO]\tCoordinates: " + str(constellations[0]))
                        print("------------------------------------")
                        print("[WARN]\tObject already found! Breaking out of mask interval loop!")
                        break

                if detected_constellations != 0:
                    print("[WARN]\tObject already found! Breaking out of percision interval loop!")
                    break


                percision = round(percision - 0.01, 2)
    cascade = None
    return detected_constellations

def main(argv):
    images_dir = ''
    mask_size_min = ''
    mask_size_max = ''
    output_name = ''
    percision_min = ''
    percision_max = ''
    cascade = ''
    scale = ''
    min_nghb = ''
    json_path = ''
    plot_images = ''
    streach_image = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "masksizeMin=", "masksizeMax=", "outputname=", "percisionMin=", "percisionMax=", "cascade=", "scale=", "minNghb=", "json=", "plot=", "streach="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--masksizeMin"):
            mask_size_min = float(arg)
        elif opt in ("--masksizeMax"):
            mask_size_max = float(arg)
        elif opt in ("--outputname"):
            output_name = arg
        elif opt in ("--percisionMin"):
            percision_min = float(arg)
        elif opt in ("--percisionMax"):
            percision_max = float(arg)
        elif opt in ("--cascade"):
            cascade = arg
        elif opt in ("--scale"):
            scale = arg
        elif opt in ("--minNghb"):
            min_nghb = arg
        elif opt in ("--json"):
            json_path = arg
        elif opt in ("--plot"):
            plot_images = int(arg)
        elif opt in ("--streach"):
            streach_image = int(arg)

    start_time = datetime.now()

    # Call the HAAR Detection function
    output = images_dir + 'output_haar/'

    if not os.path.exists(output) != False:
        print("[INFO]\tCreating directory: " + output)
        os.mkdir(output)

    if percision_max < percision_min:
        print("[ERROR]\tBad percision interval!")
        exit()

    if mask_size_max < mask_size_min:
        print("[ERROR]\tBad mask size interval!")
        exit()

    print("------------------------------------")
    print("Starting HAAR Detector")

    for file in os.listdir(cascade):
        if file.endswith(".xml"):
            cascade_path = str(cascade) + str(file)
            cascade_name = file.split("/")[-1].split(".")[0]
            print("------------------------------------")
            print("Trying to detect object: " + str(cascade_name.upper()))
            print("------------------------------------")

            detected_constellations = haarDetection(images_dir, output_name, mask_size_min, mask_size_max, percision_min, percision_max, output, cascade_path, scale, min_nghb, json_path, cascade_name, plot_images, False)

            if streach_image != 0 and detected_constellations == 0:
                print("[WARN]\tThe object maybe covers most of the image. Streaching image and trying again.")
                haarDetection(images_dir, output_name, mask_size_min, mask_size_max, percision_min, percision_max, output, cascade_path, scale, min_nghb, json_path, cascade_name, plot_images, True)

    
    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])