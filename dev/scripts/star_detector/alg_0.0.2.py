import os
import cv2
import numpy as np

# USED RGB COLORS
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RBG_GREEN = (0, 255, 0)

# USER BRIGHTNESS, CONTRAST AND GAMMA VALUES
BRIGHTNESS = 20
CONTRAST = 2.0
GAMMA = 0.4

# USER BRIGHTNESS, CONTRAST AND GAMMA VALUES
MIN_CONTOUR_AREA_PRECISION = 0.1

# GAMMA CORRECTION
def gammaCorrection(img, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(img, table)



file = 'img/ursa_major/ursa_major_001.jpg'

# READ IMAGE (RGB and BW)
img_rgb = cv2.imread(file)

# FIX IMAGE CONTRAST AND BRIGHTNESS
adjusted = cv2.convertScaleAbs(img_rgb, alpha=CONTRAST, beta=BRIGHTNESS)

# FIX IMAGE GAMMA
adjusted = gammaCorrection(adjusted, GAMMA)

# CONVERT IMAGE TO BW
img_bw = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

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

# CREATE AN IMAGE CONTAINING THE CONTOURS
img_new = cv2.bitwise_not(np.zeros(img_bw.shape, dtype=np.uint8))
img_contours = cv2.drawContours(img_new, trimmed_contours, -1, RBG_GREEN, 2)

# APPLY MARKS ON THE CONTOURS
mark_path = 'img/mark.png'
mark = cv2.imread(mark_path, 0)

rows, cols = img_contours.shape
new_rows = int((1000 * cols) / rows)
dimensions = (new_rows, 1000)
img_contours = cv2.resize(img_contours, dimensions)

thresh = cv2.threshold(img_contours, 220, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in cnts:
    cv2.drawContours(img_contours, [contour], -1, RBG_GREEN, 2)
    x, y, w, h = cv2.boundingRect(contour)
    img_contours[y: y + w*1, x: x + w*1] = cv2.resize(mark, (np.abs(x - (x + w*1)), np.abs(y - ( y + w*1))))

# SAVE FINAL IMAGE
cv2.imwrite("output/img/ursa_major_003_AI.jpg", img_contours)
cv2.waitKey(0)


