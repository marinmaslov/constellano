#!/usr/bin/env python
""" Constellano Star Detector
Python script for finding biggest and brightest stars in images and overlaying a mask over them.

Command format:
    py preparing_samples.py --images <dir> --masksize <size_in_percentage> --outputname <name> --percision <percision>
"""

import sys
import os
import cv2
import math
import time
import getopt
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "3.0.0"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py resizer.py --images <images_dir> --masksize <size_in_percentage> --outputname <name> --percision <percision>"

# Constants --------------------------------------- START
# USED RGB COLORS
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RBG_GREEN = (0, 255, 0)

# USED BRIGHTNESS, CONTRAST AND GAMMA VALUES
BRIGHTNESS = 20
CONTRAST = 1.0
GAMMA = 0.2
# Constants --------------------------------------- END

# Function for showing images
def plotImage(img):
    plt.imshow(img)
    plt.title('Image')
    plt.show()

# Function for gamma correction
def gammaCorrection(img, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(img, table)

# Function for applying contrast, brightness, gamma corrections and denoising
def imageCorrections(img):
    print("[INFO]\tApplying contrast, brightness and gamma corrections, and denoising image.")
    img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=BRIGHTNESS)
    img = gammaCorrection(img, GAMMA)
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 2, 2)
    return img

# Function for image resizing
def resizeImage(img_rgb, img_bw):
    # Fetch current dimensions
    rows, cols = img_bw.shape

    # Calculate new height
    new_cols = int((1000 * cols) / rows)

    # Defining new dimensions
    dimensions = (new_cols, 1000)

    print("[INFO]\tResizing image from: (" + str(rows) + ", " + str(cols) + ") to (" + str(1000) + ", " + str(new_cols) + ").")
    return cv2.resize(img_rgb, dimensions), cv2.resize(img_bw, dimensions)

# Function for finding the greates contour
def findGreatestContour(contours):
    print("[INFO]\tDetecting brightest star.")
    contours_areas = []
    for contour in contours:
        contours_areas.append(cv2.contourArea(contour))
    return contours_areas

# Function for finding the min and max contour
def findMinMaxContours(contours_areas, percision):
    print("[INFO]\tDefining minimum star size.")
    contours_areas.sort(reverse=True)
    max_contour_area = contours_areas[0]
    min_contour_area = float(percision) * float(max_contour_area)

    return min_contour_area, max_contour_area

# Function for trimming contours by MIN-MAX area
def trimContoursByMinMaxArea(contours, min_contour_area, max_contour_area):
    print("[INFO]\tTrimming detected star list.")
    trimmed_contours = []
    trimmed_contours_areas = []
    for contour in contours:
        if min_contour_area <= float(cv2.contourArea(contour)) <= max_contour_area:
            trimmed_contours_areas.append(cv2.contourArea(contour))
            trimmed_contours.append(contour)

    # THERE SHOULD BE A MINIMUM OF 5 STARS
    if len(trimmed_contours) < 5:
        print("[INFO]\tNumber of stars should not be below 5. Adding back the biggest from the removed ones.")
        while len(trimmed_contours) <= 5:
            min_contour_area = round(min_contour_area - 0.1, 2)
            trimmed_contours = []
            trimmed_contours_areas = []
            for contour in contours:
                if min_contour_area <= cv2.contourArea(contour) <= max_contour_area:
                    trimmed_contours_areas.append(cv2.contourArea(contour))
                    trimmed_contours.append(contour)

    return trimmed_contours

# Function for star mask preparation
def prepareStarMask(mask_size, cols):
    print("[INFO]\tSelecting dimensions for the mark based on image dimensions.")

    current_file_name = __file__.replace("\\", "/").split("/")[-1]
    current_file_parent_directory = __file__.replace(current_file_name, "")
    mark_path = current_file_parent_directory + '/img/target.png'
    mark = cv2.imread(mark_path)
    mark_dimensions = int(math.ceil(cols * float(mask_size)))

    print("[INFO]\tMark dimensions selected: (" + str(mark_dimensions) + ", " + str(mark_dimensions) + ").")

    return cv2.resize(mark, (mark_dimensions, mark_dimensions))

# Function for image overlay
def overlayImages(roi, mark_cropped):
    # mark (mask and inverse mask)
    img2gray = cv2.cvtColor(mark_cropped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(mark_cropped, mark_cropped, mask=mask)

    return cv2.add(img1_bg, img2_fg)


# Function for star mask applyment
def applyMask(file, new_file_name, trimmed_contours, mask, img_rgb, cols, rows):
    print("[INFO]\tDetected: " + str(len(trimmed_contours)) + " in file: " + str(file) + " (saving output to: " + new_file_name + ")")
    for trimmed_cnt in trimmed_contours:
        # x_contour and y_contour represent the contour center which refers to its position in the big image
        x, y, w, h = cv2.boundingRect(trimmed_cnt)

        x_contour = int(round((x + w) / 2))
        y_contour = int(round((y + h) / 2))

        x_big_image = int(cols)
        y_big_image = int(rows)

        mask_dimensions = mask.shape[0]

        x_mark_start = int(x_contour - mask_dimensions / 2)
        y_mark_start = int(y_contour - mask_dimensions / 2)

        x_mark_end = int(x_contour + mask_dimensions / 2)
        y_mark_end = int(y_contour + mask_dimensions / 2)

        # DECISION TREE
        # CHECK: Is the contour on any image border?
        if x_mark_start < 0 or y_mark_start < 0 or x_mark_end > x_big_image or y_mark_end > y_big_image:
            # CHECK: Is the contour in any corner?
            if (x_mark_start < 0 and y_mark_start < 0) or (x_mark_end > x_big_image and y_mark_end > y_big_image) or (x_mark_start < 0 and y_mark_end > y_big_image) or (x_mark_end > x_big_image and y_mark_start < 0):
                # CHECK: Is the corner on the main diagonal?
                if (x_mark_start < 0 and y_mark_start < 0) or (x_mark_end > x_big_image and y_mark_end > y_big_image):
                    # CHECK: Is the corner TOP-LEFT?
                    if (x_mark_start < 0 and y_mark_start < 0):
                        mask_cropped = mask[int(mask_dimensions - (mask_dimensions / 2 + y_contour)) : mask_dimensions, int(mask_dimensions - (mask_dimensions / 2 + x_contour)) : mask_dimensions]
                        img_rgb[0 : y_mark_end, 0 : x_mark_end] = overlayImages(img_rgb[0 : y_mark_end, 0 : x_mark_end], mask_cropped)
                        print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in TOP-LEFT corner. Applying mask with corner fix.")
                    # The corner is BOTTOM-RIGHT
                    else:
                        mask_cropped = mask[0 : int(mask_dimensions / 2 + (y_big_image - y_contour)), 0 : int(mask_dimensions / 2 + (x_big_image - x_contour))]
                        img_rgb[y_mark_start : y_big_image, x_mark_start : x_big_image] = overlayImages(img_rgb[y_mark_start : y_big_image, x_mark_start : x_big_image], mask_cropped)
                        print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in BOTTOM-RIGHT corner. Applying mask with corner fix.")
                # The contour is on the secondary diagonal
                else:
                    # CHECK: Is the corner TOP-RIGHT?
                    if (y_mark_start < 0 and x_mark_end > x_big_image):
                        mask_cropped = mask[int(mask_dimensions - (mask_dimensions / 2 + y_contour)) : mask_dimensions, 0 : int((mask_dimensions / 2) + (x_big_image - x_contour))]
                        img_rgb[0 : y_mark_end, x_mark_start : x_big_image] = overlayImages(img_rgb[0 : y_mark_end, x_mark_start : x_big_image], mask_cropped)
                        print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in TOP-RIGHT corner. Applying mask with corner fix.")
                    # The croner is BOTTOM-LEFT
                    else:
                        mask_cropped = mask[0 : int((y_big_image - y_contour) + (mask_dimensions / 2)), int(mask_dimensions - (mask_dimensions / 2 + x_contour)) : mask_dimensions]
                        img_rgb[y_mark_start : y_big_image, 0 : x_mark_end] = overlayImages(img_rgb[y_mark_start : y_big_image, 0 : x_mark_end], mask_cropped)
                        print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in BOTTOM-LEFT corner. Applying mask with corner fix.")
            # The contour is not in any corner, which means that is just on one of the borders
            else:
                # CHECK: Is the contour on the starting (top-left) borders?
                if (x_mark_start < 0 or y_mark_start < 0):
                    # CHECK: Is the contour on the left border?
                    if x_mark_start < 0:
                        mask_cropped = mask[0 : mask_dimensions, int(mask_dimensions - (mask_dimensions / 2 + x_contour)) : mask_dimensions]
                        img_rgb[y_mark_start : y_mark_end, 0 : x_mark_end] = overlayImages(img_rgb[y_mark_start : y_mark_end, 0 : x_mark_end], mask_cropped)
                        print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " on the LEFT border. Applying mask with border fix.")
                    # The contour is on the top border
                    else:
                        mask_cropped = mask[int(mask_dimensions - (mask_dimensions / 2 + y_contour)) : mask_dimensions, 0 : mask_dimensions]
                        img_rgb[0 : y_mark_end, x_mark_start : x_mark_end] = overlayImages(img_rgb[0 : y_mark_end, x_mark_start : x_mark_end], mask_cropped)
                        print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " on the TOP border. Applying mask with border fix.")
                # The contour is on the ending (bottom-right) borders
                else:
                    # CHECK: Is the contour on the right border?
                    if x_mark_end > x_big_image:
                        mask_cropped = mask[0 : mask_dimensions, 0 : int(mask_dimensions / 2 + (x_big_image - x_contour))]
                        img_rgb[y_mark_start : y_mark_end, x_mark_start : x_big_image] = overlayImages(img_rgb[y_mark_start : y_mark_end, x_mark_start : x_big_image], mask_cropped)
                        print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " on the RIGHT border. Applying mask with border fix.")
                    # The contour is on the bottom border
                    else:
                        mask_cropped = mask[0 : int(mask_dimensions / 2 + (y_big_image - y_contour)), 0 : mask_dimensions]
                        img_rgb[y_mark_start : y_big_image, x_mark_start : x_mark_end] = overlayImages(img_rgb[y_mark_start : y_big_image, x_mark_start : x_mark_end], mask_cropped)
                        print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " on the BOTTOM border. Applying mask with border fix.")
        # The contour is in not on the border, no further preparation is needed
        else:
            img_rgb[y_mark_start: y_mark_end, x_mark_start: x_mark_end] = overlayImages(img_rgb[y_mark_start: y_mark_end, x_mark_start: x_mark_end], mask_cropped)
            print("[INFO]\tDetected star of size: " + str(cv2.contourArea(trimmed_cnt)) + " in image. Applying mask.")

def detectStars(images_dir, output_name, mask_size, percision, file, output, counter):
    # PREPARE OUTPUT NAME
    zeros = "00000"
    zeros_counter = len(str(counter))
    while zeros_counter > 0:
        zeros = zeros[:-1]
        zeros_counter = zeros_counter - 1
    new_file_name = str(output + str(output_name) + "_" + str(zeros) + str(counter) + ".jpg")

    # READ IMAGE (RGB and BW)
    print("[INFO]\tReading: " + file)
    img_rgb = cv2.imread(images_dir + file)

    # CONTRAST, BRIGHTNESS AND GAMMA CORRECTIONS, AND DENOISING
    img_adjusted = imageCorrections(img_rgb)

    # CONVERT IMAGE TO BW
    print("[INFO]\tConverting image to grayscale.")
    img_bw = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2GRAY)
    #plotImage(img_bw)

    # RESIZE IMAGE
    img_rgb_resized, img_bw_resized = resizeImage(img_rgb, img_bw);

    # CREATE THRASHOLD IMAGE
    print("[INFO]\tCreating thresholds.")
    thresholds = cv2.threshold(img_bw_resized, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #plotImage(thresholds)

    # FIND CONTURES (STARS)
    print("[INFO]\tDetecting stars.")
    contours = cv2.findContours(thresholds, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # FIND THE GREATEST CONTOUR AREA
    contours_areas = findGreatestContour(contours)

    # DEFINE MIN-MAX CONTOUR AREA
    min_contour_area, max_contour_area = findMinMaxContours(contours_areas, percision)

    # TRIM CONTOURS BY MIN-MAX CONTOUR AREA
    trimmed_contours = trimContoursByMinMaxArea(contours, min_contour_area, max_contour_area)

    # PREPARE THE STAR MASK
    mask = prepareStarMask(mask_size, img_bw_resized.shape[0])

    # APPLY THE STAR MASK
    img_rgb = applyMask(file, new_file_name, trimmed_contours, mask, img_rgb, img_rgb.shape[0], img_rgb.shape[1])

    return img_rgb, new_file_name, img_rgb_resized

# Overall star detection function
def starDetector(images_dir, output_name, mask_size, percision):
    start_time = time.time()
    output = images_dir + 'output/'

    if not os.path.exists(output) != False:
        print("[INFO]\tCreating directory: " + output)
        os.mkdir(output)

    counter = 0
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            img_rgb, new_file_name = detectStars(images_dir, output_name, mask_size, percision, file, output, counter)
            # SAVE THE FINAL IMAGE
            print("[INFO]\tSaving image: " + str(new_file_name))
            cv2.imwrite(new_file_name, img_rgb)
            counter = counter + 1
    print("------------------------------------")
    print("[INFO]\tTotal files created:" + str(counter))
    print("[INFO]\tTotal execution time: " + str(time.strftime('%H:%M:%S', (time.time() - start_time))) + ".")

def main(argv):
    images_dir = ''
    mask_size = ''
    output_name = ''
    percision = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["images=", "masksize=", "outputname=", "percision="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--markersize"):
            mask_size = arg
        elif opt in ("--outputname"):
            output_name = arg
        elif opt in ("--percision"):
            percision = arg

    # Call the StarDetector function
    starDetector(images_dir, output_name, mask_size, percision)

if __name__ == "__main__":
    main(sys.argv[1:])