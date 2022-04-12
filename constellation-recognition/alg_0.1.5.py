import os
from os import path
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


# Constants --------------------------------------- START
# USED RGB COLORS
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RBG_GREEN = (0, 255, 0)

# USED BRIGHTNESS, CONTRAST AND GAMMA VALUES
BRIGHTNESS = 20
CONTRAST = 2.0
GAMMA = 0.2

# USED BRIGHTNESS, CONTRAST AND GAMMA VALUES
MIN_CONTOUR_AREA_PRECISION = 0.18

# USED MARK SIZE
MARK_SIZE_PERCENTAGE = 0.05
# Constants --------------------------------------- END

# Functions --------------------------------------- START
# GAMMA CORRECTION
def gammaCorrection(img, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(img, table)
# Functions --------------------------------------- END

# Algorithm --------------------------------------- START
# READ IMAGE (RGB and BW)
img_rgb = cv2.imread("img/ursa_major/ursa_major_001.jpg")

# CONTRAST, BRIGHTNESS AND GAMMA CORRECTIONS
adjusted = cv2.convertScaleAbs(img_rgb, alpha=CONTRAST, beta=BRIGHTNESS)
adjusted = gammaCorrection(adjusted, GAMMA)

# CONVERT IMAGE TO BW
img_bw = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

#plt.imshow(img_bw, 'gray')
#plt.title('stars')
#plt.show()

# RESIZE IMAGE
rows, cols = img_bw.shape
new_cols = int((1000 * cols) / rows)
dimensions = (new_cols, 1000)
img_bw_resized = cv2.resize(img_bw, dimensions)

# CREATE THRASHOLDS (STARS)
thresholds = cv2.threshold(img_bw_resized, 220, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

# FIND CONTURES (STARS)
contours = cv2.findContours(thresholds, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

# FIND THE GREATEST CONTOUR AREA
contours_areas = []
for contour in contours:
    contours_areas.append(cv2.contourArea(contour))

# DEFINE MIN-MAX CONTOUR AREA
contours_areas.sort(reverse = True)
max_contour_area = contours_areas[0]
min_contour_area = MIN_CONTOUR_AREA_PRECISION * max_contour_area

# TRIM CONTOURS BY MIN-MAX CONTOUR AREA
trimmed_contours = []
trimmed_contours_areas = []
for contour in contours:
    if min_contour_area <= cv2.contourArea(contour) <= max_contour_area:
        trimmed_contours_areas.append(cv2.contourArea(contour))
        trimmed_contours.append(contour)

# SELECT DIMENSIONS FOR THE MARK
mark_path = 'img/mark.png'
mark = cv2.imread(mark_path, 0)
mark_dimensions = int(math.ceil(dimensions[1] * MARK_SIZE_PERCENTAGE))
mark_dimensions_offset = int(math.ceil(mark_dimensions/2))

print("Number of stars detected: ", len(trimmed_contours))

for trimmed_cnt in trimmed_contours:
    x, y, w, h = cv2.boundingRect(trimmed_cnt)

    x_offset = x - mark_dimensions_offset
    y_offset = y - mark_dimensions_offset

    x_end = x_offset + mark_dimensions
    y_end = y_offset + mark_dimensions

    mark_resized = cv2.resize(mark, (mark_dimensions, mark_dimensions))

    """
    THERE ARE 6 POSSIBLE SCENARIOS FOR OVERFLOW

    (ACTUALLY THERE ARE MORE, BUT THE OPTIONS WHERE THE MARK WILL OVERFLOW THE WHOLE,
    OR AT LAST GREATER PARTS OF THE IMAGE CAN BE IGNORED AS THE MARK SIZE IS:
    MARK_SIZE_PERCENTAGE × 'the greater dimension of the image')

    1. At the START for x OR y                  - NAPISANO
    2. At the END for x OR y                    - NAPISANO
    3. At the START for x AND y
    4. At the END for x AND y
    5. At the START for x AND END for y
    6. At the END for x AND START for y
    """

    # 3. At the START for x AND y
    if x_offset < 0 and y_offset < 0:
        mark_cropped = mark_resized[abs(y_offset) : mark_dimensions, abs(x_offset) : mark_dimensions]
        img_bw_resized[0 : y_end, 0 : x_end] = mark_cropped
        
    # 4. At the END for x AND y
    elif x_end > dimensions[0] and y_end > dimensions[1]:
        mark_cropped = mark_resized[0 : mark_dimensions - (y_end - dimensions[1]), 0 : mark_dimensions - (x_end - dimensions[0])]
        img_bw_resized[y_offset : dimensions[1], x_offset : dimensions[0]] = mark_cropped
        
    # 5. and 6.
    elif (x_offset < 0 or y_offset < 0) and (x_end > dimensions[0] or y_end > dimensions[1]):
        # 5. At the START for x AND END for y
        if x_offset < 0 and y_end > dimensions[1]:
            mark_cropped = mark_resized[0 : mark_dimensions - (y_end - dimensions[1]), abs(x_offset) : mark_dimensions]
            img_bw_resized[y_offset : dimensions[1], 0 : x_end] = mark_cropped

        # 6. At the END for x AND START for y
        if y_offset < 0 and x_end > dimensions[0]:
            mark_cropped = mark_resized[abs(y_offset) : mark_dimensions, 0 : mark_dimensions - (x_end - dimensions[0])]
            img_bw_resized[0 : y_end, x_offset : dimensions[0]] = mark_cropped
    # 2. At the END for x OR y  POPRAVI KOMENTARE
    elif (x_end > dimensions[0] or y_end > dimensions[1]) and not (x_offset < 0 or y_offset < 0):
        if x_end > dimensions[0] or y_end > dimensions[1]:
            if x_end > dimensions[0]:
                mark_cropped = mark_resized[0:mark_dimensions, 0: mark_dimensions - (x_end - dimensions[0])]
                img_bw_resized[y_offset : y_end, x_offset : dimensions[0]] = mark_cropped
            if y_end > dimensions[1]:
                mark_cropped = mark_resized[0:mark_dimensions - (y_end - dimensions[1]), 0:mark_dimensions]
                img_bw_resized[y_offset : dimensions[1], x_offset : x_end] = mark_cropped  
    # 2. At the END for x OR y   POPRAVI KOMENTARE
    elif (x_offset < 0 or y_offset < 0) and not (x_end > dimensions[0] or y_end > dimensions[1]):         
        if x_offset < 0 or y_offset < 0:
            if x_offset < 0:
                mark_cropped = mark_resized[0:mark_dimensions, abs(x_offset):mark_dimensions]
                img_bw_resized[y_offset : y_end, 0 : x_end] = mark_cropped
            if y_offset < 0:
                mark_cropped = mark_resized[abs(y_offset):mark_dimensions, 0:mark_dimensions]
                img_bw_resized[0 : y_end, x_offset : x_end] = mark_cropped    
    # Anything else (not on edges)
    else:
        img_bw_resized[y_offset : y_end, x_offset : x_end] = cv2.resize(mark, (mark_dimensions, mark_dimensions))

# SAVE FINAL IMAGE
new_file_name = "ursa_major_001_AI.jpg"
cv2.imwrite(new_file_name, img_bw_resized)
cv2.waitKey(0)
# Algorithm --------------------------------------- END