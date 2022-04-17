import os
from os import path
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# USED RGB COLORS
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RBG_GREEN = (0, 255, 0)

# USED BRIGHTNESS, CONTRAST AND GAMMA VALUES
BRIGHTNESS = 20
CONTRAST = 2.0
GAMMA = 0.2

# USED BRIGHTNESS, CONTRAST AND GAMMA VALUES
MIN_CONTOUR_AREA_PRECISION = 0.1

# USED MARK SIZE
MARK_SIZE_PERCENTAGE = 0.05

# GAMMA CORRECTION
def gammaCorrection(img, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(img, table)



location = 'C:/Users/easmsma/Desktop/Diplomski/constellation-recognition/constellation-recognition/img/ursa_major/'
output = location + 'outout/'

print(os.path.exists(output))

if not os.path.exists(output) != False:
    os.mkdir(output)

counter = 0

for file in os.listdir(location):
    if file.endswith(".jpg"):
        print(file)
        # READ IMAGE (RGB and BW)
        img_rgb = cv2.imread(location + file)

        # FIX IMAGE CONTRAST AND BRIGHTNESS
        adjusted = cv2.convertScaleAbs(img_rgb, alpha=CONTRAST, beta=BRIGHTNESS)

        # FIX IMAGE GAMMA
        adjusted = gammaCorrection(adjusted, GAMMA)

        # CONVERT IMAGE TO BW
        img_bw = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

        #plt.imshow(img_bw, 'gray')
        #plt.title('stars')
        #plt.show()

        # CREATE THRASHOLDS (STARS)
        thresholds = cv2.threshold(img_bw, 220, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

        # FIND CONTURES (STARS)
        contours = cv2.findContours(thresholds, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # FIND THE GREATEST CONTOUR AREA
        contours_areas = []
        for contour in contours:
            contours_areas.append(cv2.contourArea(contour))

        contours_areas.sort(reverse = True)
        max_contour_area = contours_areas[0]
        min_contour_area = MIN_CONTOUR_AREA_PRECISION * max_contour_area

        # TRIM CONTOURS BY AREA SIZE
        trimmed_contours = []
        trimmed_contours_areas = []
        for contour in contours:
            if min_contour_area <= cv2.contourArea(contour) <= max_contour_area:
                trimmed_contours_areas.append(cv2.contourArea(contour))
                trimmed_contours.append(contour)

        # CREATE AN NEW EMPTY IMAGE
        img_new = np.zeros(img_bw.shape, dtype=np.uint8) #cv2.bitwise_not()
        img_contours = cv2.drawContours(img_new, trimmed_contours, -1, RBG_GREEN, -1)

        #plt.imshow(img_contours, 'gray')
        #plt.title('stars')
        #plt.show()

        # APPLY MARKS ON THE CONTOURS
        mark_path = 'img/mark.png'
        mark = cv2.imread(mark_path, 1)

        rows, cols = img_contours.shape

        new_cols = int((1000 * cols) / rows)
        print(new_cols, 1000)
        dimensions = (new_cols, 1000)
        # THE IMAGE TO RUN CONTOURS ONCE MORE
        img_contours = cv2.resize(img_contours, dimensions)
        
        # THE FINAL IMAGE TO APPLY MARKS
        adjusted = cv2.resize(img_bw, dimensions)

        # SELECT DIMENSIONS FOR THE MARK
        mark_dimensions = int(math.ceil(dimensions[1]*MARK_SIZE_PERCENTAGE))
        mark_dimensions_offset = int(math.ceil(mark_dimensions/2))

        # CONVERT TO BW
        #img_contours = cv2.cvtColor(img_contours, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(img_contours, 220, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        print("Number of stars detected: ", len(cnts))

        for cnt in cnts:
            #cv2.drawContours(adjusted, [cnt], -1, RBG_GREEN, 2)
            x, y, w, h = cv2.boundingRect(cnt)

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
                adjusted[0 : y_end, 0 : x_end] = mark_cropped
                break
            # 4. At the END for x AND y
            elif x_end > dimensions[0] and y_end > dimensions[1]:
                mark_cropped = mark_resized[0 : mark_dimensions - (y_end - dimensions[1]), 0 : mark_dimensions - (x_end - dimensions[0])]
                adjusted[y_offset : dimensions[1], x_offset : dimensions[0]] = mark_cropped
                break
            # 5. and 6.
            elif (x_offset < 0 or y_offset < 0) and (x_end > dimensions[0] or y_end > dimensions[1]):
                # 5. At the START for x AND END for y
                if x_offset < 0 and y_end > dimensions[1]:
                    mark_cropped = mark_resized[0 : mark_dimensions - (y_end - dimensions[1]), abs(x_offset) : mark_dimensions]
                    adjusted[y_offset : dimensions[1], 0 : x_end] = mark_cropped

                # 6. At the END for x AND START for y
                if y_offset < 0 and x_end > dimensions[0]:
                    mark_cropped = mark_resized[abs(y_offset) : mark_dimensions, 0 : mark_dimensions - (x_end - dimensions[0])]
                    adjusted[0 : y_end, x_offset : dimensions[0]] = mark_cropped
                break
            # 2. At the END for x OR y  
            elif (x_end > dimensions[0] or y_end > dimensions[1]) and not (x_offset < 0 or y_offset < 0):
                if x_end > dimensions[0] or y_end > dimensions[1]:
                    #OVA TRIBA POPRAVIT!
                    if x_end > dimensions[0]:
                        mark_cropped = mark_resized[0:mark_dimensions, 0: mark_dimensions - (x_end - dimensions[0])]
                        adjusted[y_offset : y_end, x_offset : dimensions[0]] = mark_cropped
                    if y_end > dimensions[1]:
                        mark_cropped = mark_resized[0:mark_dimensions - (y_end - dimensions[1]), 0:mark_dimensions]
                        adjusted[y_offset : dimensions[1], x_offset : x_end] = mark_cropped  
                break
            elif (x_offset < 0 or y_offset < 0) and not (x_end > dimensions[0] or y_end > dimensions[1]):         
                if x_offset < 0 or y_offset < 0:
                    if x_offset < 0:
                        mark_cropped = mark_resized[0:mark_dimensions, abs(x_offset):mark_dimensions]
                        adjusted[y_offset : y_end, 0 : x_end] = mark_cropped
                    if y_offset < 0:
                        mark_cropped = mark_resized[abs(y_offset):mark_dimensions, 0:mark_dimensions]
                        adjusted[0 : y_end, x_offset : x_end] = mark_cropped    
                break
            # Anything else (not on edges)
            else:
                #adjusted[y : y - mark_dimensions_offset + mark_dimensions, x : x - mark_dimensions_offset  + mark_dimensions] = cv2.resize(mark, (mark_dimensions, mark_dimensions))
                adjusted[y_offset : y_end, x_offset : x_end] = cv2.resize(mark, (mark_dimensions, mark_dimensions))

        # SAVE FINAL IMAGE
        new_file_name = str(output + os.path.splitext(file)[0] + "_AI.jpg");
        print("Creating: " + str(new_file_name))
        cv2.imwrite(new_file_name, adjusted)
        cv2.waitKey(0)
        counter = counter + 1
print("Total files created: " + str(counter)), 

