#!/usr/bin/env python
""" Constellano Star Recognition
Python script for finding biggest and brightest stars in images and overlaying a target over them.

Command format:
    py star_detector.py <image_dir> <min_contour_area_percision>

Command example:
    py preparing_samples.py img/ 0.18
"""

import sys
import os
import cv2
import numpy as np
import math
import time
import getopt
import json

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.0.2"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py resizer.py --images <images_dir> --percision <percision> --log <log_level>"

# Constants --------------------------------------- START
# USED RGB COLORS
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RBG_GREEN = (0, 255, 0)

# USED BRIGHTNESS, CONTRAST AND GAMMA VALUES
BRIGHTNESS = 20
CONTRAST = 1.0
GAMMA = 0.2

# USE BLUR
# UNSET_MINIMUM_STARS = 0
# if(len(sys.argv) > 3):
#     UNSET_MINIMUM_STARS = int(sys.argv[3])

# USED MARK SIZE
MARK_SIZE_PERCENTAGE = 0.08
# Constants --------------------------------------- END

# Functions --------------------------------------- START
# GAMMA CORRECTION
def gammaCorrection(img, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(img, table)

# IMAGE OVERLAY
def overlayImages(roi, mark_cropped):
    # mark (mask and inverse mask)
    img2gray = cv2.cvtColor(mark_cropped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(mark_cropped, mark_cropped, mask=mask)

    return cv2.add(img1_bg, img2_fg)
# Functions --------------------------------------- END


# Algorithm --------------------------------------- START
def main(argv):
    images_dir = ''
    percision = ''
    log_level = ''
    cascade = ''
    scale = ''
    min_nghb = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "percision=", "log=", "cascade=", "scale=", "minNghb="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--percision"):
            percision = arg
        elif opt in ("--log"):
            log_level = arg
        elif opt in ("--cascade"):
            cascade = arg
        elif opt in ("--scale"):
            scale = arg
        elif opt in ("--minNghb"):
            min_nghb = arg

    #logging.basicConfig(filename=images_dir + 'output.log', encoding='utf-8', level=log_level.upper())

    output = images_dir + 'output_haar/'

    if not os.path.exists(output) != False:
        print("\033[2;32;40m[INFO]\033[0;0m" + "\tCreating directory: " + output)
        os.mkdir(output)

    cascade = cv2.CascadeClassifier(cascade)
    constellations_data = json.load(open('constellations.json'))
    lyra_data = (constellations_data["constellations"])["lyra"]
    lyra_star_list_data = (lyra_data)["star-list"]
    lyra_stars_data = (lyra_star_list_data)["stars"]
    lyra_connections_data = (lyra_star_list_data)["connections"]
    print(lyra_stars_data)
    print(lyra_connections_data)
    counter = 0
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            # PREPARE OUTPUT NAME
            zeros = "00000"
            zeros_counter = len(str(counter))
            while zeros_counter > 0:
                zeros = zeros[:-1]
                zeros_counter = zeros_counter - 1

            new_file_name = str(output + "detected_" + str(zeros) + str(counter) + ".jpg")

            # READ IMAGE (RGB and BW)
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tOpening: " + file)
            img_rgb = cv2.imread(images_dir + file)

            # CONTRAST, BRIGHTNESS AND GAMMA CORRECTIONS
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tApplying contrast, brightness and gamma corrections.")
            adjusted = cv2.convertScaleAbs(img_rgb, alpha=CONTRAST, beta=BRIGHTNESS)
            adjusted = gammaCorrection(adjusted, GAMMA)

            # DENOISING
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDenoising image.")
            adjusted = cv2.fastNlMeansDenoisingColored(adjusted, None, 5, 5, 2, 2)

            #if USE_BLUR == 1:
            #    ksize = (1, 1)
            #    adjusted = cv2.blur(adjusted, ksize)

            # CONVERT IMAGE TO BW
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tConverting image to grayscale.")
            img_bw = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

            #plt.imshow(img_bw, 'gray')
            #plt.title('stars')
            #plt.show()

            # RESIZE IMAGE
            rows, cols = img_bw.shape
            new_cols = int((1000 * cols) / rows)
            dimensions = (new_cols, 1000)
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tResizing image from: (" + str(rows) + ", " + str(cols) + ") to (" + str(1000) + ", " + str(new_cols) + ").")
            img_bw_resized = cv2.resize(img_bw, dimensions)
            img_rgb_resized = cv2.resize(img_rgb, dimensions)

            # CREATE THRASHOLDS (STARS)
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tCreating thresholds.")
            thresholds = cv2.threshold(img_bw_resized, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            #cv2.imshow("Input", thresholds)

            # FIND CONTURES (STARS)
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetecting stars.")
            contours = cv2.findContours(thresholds, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

            # FIND THE GREATEST CONTOUR AREA
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetecting biggest star.")
            contours_areas = []
            for contour in contours:
                contours_areas.append(cv2.contourArea(contour))

            # DEFINE MIN-MAX CONTOUR AREA
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDefining minimum star size.")
            contours_areas.sort(reverse=True)
            max_contour_area = contours_areas[0]
            min_contour_area = float(percision) * float(max_contour_area)  # * len(contours_areas) #
            #print("MAX: " + str(max_contour_area) + " MIN: " + str(min_contour_area))

            # TRIM CONTOURS BY MIN-MAX CONTOUR AREA
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tTrimming detected star list.")
            trimmed_contours = []
            trimmed_contours_areas = []
            for contour in contours:
                if min_contour_area <= float(cv2.contourArea(contour)) <= max_contour_area:
                    trimmed_contours_areas.append(cv2.contourArea(contour))
                    trimmed_contours.append(contour)

            # THERE SHOULD BE A MINIMUM OF 10 STARS
            if len(trimmed_contours) < 5:
                if (log_level.upper() == "DEBUG"):
                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tNumber of stars should not be below 5. Adding back the biggest from the removed ones.")
                while len(trimmed_contours) <= 5:
                    min_contour_area = round(min_contour_area - 0.1, 2)
                    trimmed_contours = []
                    trimmed_contours_areas = []
                    for contour in contours:
                        if min_contour_area <= cv2.contourArea(contour) <= max_contour_area:
                            trimmed_contours_areas.append(cv2.contourArea(contour))
                            trimmed_contours.append(contour)

            # SELECT DIMENSIONS FOR THE MARK
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tSelecting dimensions for the mark based on image dimensions.")
            current_file_name = __file__.replace("\\", "/").split("/")[-1]
            current_file_parent_directory = __file__.replace(current_file_name, "")
            mark_path = current_file_parent_directory + '/img/target.png'
            mark = cv2.imread(mark_path)
            mark_dimensions = int(math.ceil(dimensions[1] * MARK_SIZE_PERCENTAGE))
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tMark dimensions selected: (" + str(mark_dimensions) + ", " + str(mark_dimensions) + ").")
            mark_dimensions_offset = int(math.ceil(mark_dimensions/2 - 0.15*mark_dimensions))
            mark_resized = cv2.resize(mark, (mark_dimensions, mark_dimensions))

            print("\033[2;32;40m[INFO]\033[0;0m" + "\tDetected: " + str(len(trimmed_contours)) + " in file: " + str(file) + " (saving output to: " + new_file_name + ")")

            for trimmed_cnt in trimmed_contours:
                x, y, _, _ = cv2.boundingRect(trimmed_cnt)

                x_offset = x - mark_dimensions_offset
                y_offset = y - mark_dimensions_offset

                x_end = x_offset + mark_dimensions
                y_end = y_offset + mark_dimensions

                # DECISION TREE
                # CHECK: Is the contour on any image border?
                if x_offset < 0 or y_offset < 0 or x_end > dimensions[0] or y_end > dimensions[1]:
                    # CHECK: Is the contour in any corner?
                    if (x_offset < 0 and y_offset < 0) or (x_end > dimensions[0] and y_end > dimensions[1]) or (x_offset < 0 and y_end > dimensions[1]) or (y_offset < 0 and x_end > dimensions[0]):
                        # CHECK: Is the corner on the main diagonal?
                        if (x_offset < 0 and y_offset < 0) or (x_end > dimensions[0] and y_end > dimensions[1]):
                            # CHECK: Is the corner TOP-LEFT?
                            if (x_offset < 0 and y_offset < 0):
                                mark_cropped = mark_resized[abs(y_offset): mark_dimensions, abs(x_offset): mark_dimensions]
                                img_rgb_resized[0: y_end, 0: x_end] = overlayImages(img_rgb_resized[0: y_end, 0: x_end], mark_cropped)
                                if (log_level.upper() == "DEBUG"):
                                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in TOP-LEFT corner. \033[2;36;40mApplying mask with corner fix.\033[0;0m")
            
                            # The corner is BOTTOM-RIGHT
                            else:
                                print(str(dimensions[0]) + " " + str(dimensions[1]))
                                print(str(y) + " " + str(x))
                                mark_cropped = mark_resized[0 : int(mark_dimensions/2) + dimensions[0] - y, 0 : int(mark_dimensions/2) + dimensions[1] - x]


                                #mark_cropped = mark_resized[0: dimensions[1] - y_offset - (dimensions[1] - y), 0: dimensions[0] - x_offset - (dimensions[0] - x)]
                                img_rgb_resized[y - int(mark_dimensions/2): dimensions[1], x - int(mark_dimensions/2): dimensions[0]] = overlayImages(img_rgb_resized[y - int(mark_dimensions/2): dimensions[1], x - int(mark_dimensions/2): dimensions[0]], mark_cropped)
                                if (log_level.upper() == "DEBUG"):
                                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in BOTTOM-RIGHT corner. \033[2;36;40mApplying mask with corner fix.\033[0;0m")
                        # The contour is on the secondary diagonal
                        else:
                            # CHECK: Is the corner TOP-RIGHT?
                            if (y_offset < 0 and x_end > dimensions[0]):
                                mark_cropped = mark_resized[abs(y_offset): mark_dimensions, 0: mark_dimensions - (x_end - dimensions[0])]
                                img_rgb_resized[0: y_end, x_offset: dimensions[0]] = overlayImages(img_rgb_resized[0: y_end, x_offset: dimensions[0]], mark_cropped)
                                if (log_level.upper() == "DEBUG"):
                                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in TOP-RIGHT corner. \033[2;36;40mApplying mask with corner fix.\033[0;0m")
                            # The croner is BOTTOM-LEFT
                            else:
                                mark_cropped = mark_resized[0: mark_dimensions - (y_end - dimensions[1]), abs(x_offset): mark_dimensions]
                                img_rgb_resized[y_offset: dimensions[1], 0: x_end] = overlayImages(img_rgb_resized[y_offset: dimensions[1], 0: x_end], mark_cropped)
                                if (log_level.upper() == "DEBUG"):
                                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in BOTTOM-LEFT corner. \033[2;36;40mApplying mask with corner fix.\033[0;0m")
                    # The contour is not in any corner, which means that is just on one of the borders
                    else:
                        # CHECK: Is the contour on the starting (top-left) borders?
                        if (x_offset < 0 or y_offset < 0):
                            # CHECK: Is the contour on the left border?
                            if x_offset < 0:
                                mark_cropped = mark_resized[0:mark_dimensions, abs(x_offset):mark_dimensions]
                                img_rgb_resized[y_offset: y_end, 0: x_end] = overlayImages(img_rgb_resized[y_offset: y_end, 0: x_end], mark_cropped)
                                if (log_level.upper() == "DEBUG"):
                                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " on the LEFT border. \033[2;36;40mApplying mask with border fix.\033[0;0m")
                            # The contour is on the top border
                            else:
                                mark_cropped = mark_resized[abs(y_offset):mark_dimensions, 0:mark_dimensions]
                                img_rgb_resized[0: y_end, x_offset: x_end] = overlayImages(img_rgb_resized[0: y_end, x_offset: x_end], mark_cropped)
                                if (log_level.upper() == "DEBUG"):
                                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " on the TOP border. \033[2;36;40mApplying mask with border fix.\033[0;0m")
                        # The contour is on the ending (bottom-right) borders
                        else:
                            # CHECK: Is the contour on the right border?
                            if x_end > dimensions[0]:
                                mark_cropped = mark_resized[0:mark_dimensions, 0: mark_dimensions - (x_end - dimensions[0])]
                                img_rgb_resized[y_offset: y_end, x_offset: dimensions[0]] = overlayImages(img_rgb_resized[y_offset: y_end, x_offset: dimensions[0]], mark_cropped)
                                if (log_level.upper() == "DEBUG"):
                                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " on the RIGHT border. \033[2;36;40mApplying mask with border fix.\033[0;0m")
                            # The contour is on the bottom border
                            else:
                                mark_cropped = mark_resized[0:mark_dimensions - (y_end - dimensions[1]), 0:mark_dimensions]
                                img_rgb_resized[y_offset: dimensions[1], x_offset: x_end] = overlayImages(img_rgb_resized[y_offset: dimensions[1], x_offset: x_end], mark_cropped)
                                if (log_level.upper() == "DEBUG"):
                                    print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " on the BOTTOM border. \033[2;36;40mApplying mask with border fix.\033[0;0m")
                # The contour is in not on the border, no further preparation is needed
                else:
                    img_rgb_resized[y_offset: y_end, x_offset: x_end] = overlayImages(img_rgb_resized[y_offset: y_end, x_offset: x_end], mark_resized)
                    if (log_level.upper() == "DEBUG"):
                        print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in image. \033[2;36;40mApplying mask.\033[0;0m")

            # SAVE THE FINAL IMAGE
            print("\033[2;32;40m[INFO]\033[0;0m" + "\tSaving image: " + str(new_file_name))
            #cv2.imwrite(new_file_name, img_rgb_resized)
            #cv2.waitKey(0)
            cv2.imshow("Input", img_rgb_resized)
            cv2.waitKey(0)



            haar_counter = 0
            #output = images + "haar_detection/"

            #if not os.path.exists(output) != False:
            #    print("\033[2;32;40m[INFO]\033[0;0m" + "\tCreating directory: " + output)
            #    os.mkdir(output)

            




            #new_file_name = str(output + "haar_detected_" + str(counter) + ".jpg")

            img = img_rgb_resized #cv2.imread(images + file)

            rows, cols = img_bw.shape
            new_cols = int((1000 * cols) / rows)
            dimensions = (new_cols, 1000)
            if (log_level.upper() == "DEBUG"):
                print("\033[2;35;40m[DEBUG]\033[0;0m" + "\tResizing image from: (" + str(rows) + ", " + str(cols) + ") to (" + str(1000) + ", " + str(new_cols) + ").")
            #img_bw_resized = cv2.resize(img_bw, dimensions)
            img_rgb_resized = cv2.resize(img_rgb, dimensions)

            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            constellations = cascade.detectMultiScale(gray, float(scale), int(min_nghb), flags=cv2.CASCADE_SCALE_IMAGE)

            font = cv2.FONT_HERSHEY_SIMPLEX
            for (x, y, w, h) in constellations:
                # FETCH THE CROPPED IMAGE OF THE DETECTED CONSTELLATION
                #print(str(img_rgb[y : y + h, x : x + w].shape))
                detected_constellation_img = img_rgb_resized[y : y + h, x : x + w]
                #print(str(detected_constellation_img.shape))
                cv2.imshow("Input", detected_constellation_img)
                cv2.waitKey(0)

                #cv2.rectangle(img_rgb, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 2)
                cv2.putText(img_rgb_resized, str(lyra_data["name"].title()), (x, y - 20), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

                # GET CONTOURS ON CROPPED DETECTED IMAGE
                #detected_constellation_img
                #haar_adjusted = cv2.convertScaleAbs(detected_constellation_img, alpha=CONTRAST, beta=BRIGHTNESS)
                #haar_adjusted = gammaCorrection(haar_adjusted, GAMMA)
                haar_img_bw = cv2.cvtColor(detected_constellation_img, cv2.COLOR_BGR2GRAY)
                haar_thresholds = cv2.threshold(haar_img_bw, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                haar_contours = cv2.findContours(haar_thresholds, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

                # Create contours dict
                #haar_contours_dict = {}
                #for contour in haar_contours:
                #    haar_contours_dict[cv2.contourArea(contour)] = contour

                # Create sorted list of dict values
                haar_contours_sorted = []
                for contour in haar_contours:
                    haar_contours_sorted.append(cv2.contourArea(contour))
                haar_contours_sorted.sort(reverse = True)

                # THERE SHOULD BE A MAXIMUM OF len(constellation_star_list) STARS
                trimmed_contours = []
                trimmed_contours_areas = []

                #haar_contours_dict_sorted = collections.OrderedDict(sorted(haar_contours_dict.items()))
                """
                print("CONTURE AREAS!")
                print(haar_contours_sorted)

                for i in range(0, len(haar_contours_sorted) - 1):
                    print("ZVIJEZDA: " + str(lyra_stars_data[str(i)]))
                    print("odgovara konturi: " + str(haar_contours_sorted[i]))
                """

                #cv2.line(image, start_point, end_point, color, thickness)
                

                haar_contours_sorted_coordinates = []
                star = 0
                for area in haar_contours_sorted[:len(lyra_stars_data)]:
                    for contour in haar_contours:
                        if area == cv2.contourArea(contour):
                            x1, y1, w1, h1 = cv2.boundingRect(contour)
                            haar_contours_sorted_coordinates.append((int(x1 + (w1/2)), int(y1 + (h1/2))))
                            print(str((int(x1 - 45), int(y1 - 20))))
                            cv2.putText(detected_constellation_img, str(((lyra_stars_data[str(star)])["name"].title())[:3]), (int(x1 - 45), int(y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, float(0.7), (255, 255, 255), 2)
                            """
                                Imamo Errore jer su tu i zvijezde koje nisu bas pozeljene!!!!
                            """
                            #print(str((lyra_stars_data[str(star)])["symbol"]))
                            #cv2.addText(detected_constellation_img, str((lyra_stars_data[str(star)])["symbol"]), (x1 - 10, y1 - 10), QFont("Arial", 10, QFont.Bold))
                            
                    star = star + 1

                # DRAW LINES
                for connection in lyra_connections_data:
                    cv2.line(detected_constellation_img, haar_contours_sorted_coordinates[connection[0]], haar_contours_sorted_coordinates[connection[1]], (255, 255, 255), 2)

                # WRITE NAMES
                

                #for star in lyra_stars_data:
                #    cv2.putText(detected_constellation_img, str(area), (x1, y1), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                #img_rgb[y : y + h, x : x + w] = detected_constellation_img
                img_rgb_resized[y : y + h, x : x + w] = detected_constellation_img

                #mark_cropped = mark_resized[abs(y_offset):mark_dimensions, 0:mark_dimensions]
                #img_rgb_resized[0: y_end, x_offset: x_end] = overlayImages(img_rgb_resized[0: y_end, x_offset: x_end], mark_cropped)

                #cv2.imwrite(new_file_name, detected_constellation_img)

            print("\033[2;32;40m[INFO]\033[0;0m" + "\tSaving image: " + str(new_file_name))
            cv2.imwrite(new_file_name, img_rgb_resized)
            
            haar_counter = haar_counter + 1
            print("------------------------------------")
            print("\033[2;32;40m[INFO]\033[0;0m" + "\t\033[2;44;47mTotal files created:\t" + str(counter) + "\033[0;0m")
            counter = counter + 1


    print("------------------------------------")
    print("\033[2;32;40m[INFO]\033[0;0m" + "\t\033[2;44;47mTotal files created:\t" + str(counter) + "\033[0;0m")
    # Algorithm --------------------------------------- END

if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    print("\033[2;32;40m[INFO]\033[0;0m" + "\tTotal execution time: " + str((time.time() - start_time)) + " seconds.\033[0;0m")