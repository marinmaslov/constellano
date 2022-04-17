import sys
import os
import cv2
import numpy as np


# Constants --------------------------------------- START
NEW_SIZE = 100
if len(sys.argv) > 2:
    NEW_SIZE = int(sys.argv[2])
# Constants --------------------------------------- END

# Functions --------------------------------------- START
def resize_image(img, newRows, newCols):
    return cv2.resize(img, (newCols, newRows), interpolation = cv2.INTER_AREA)

def fill(img):
    filled = np.zeros((NEW_SIZE, NEW_SIZE, 3), np.uint8)

    rows, cols, _ = img.shape
    filled_rows, filled_cols, _ = filled.shape

    if rows > cols:
        free_spaces = NEW_SIZE - cols
        for i in range(0, filled_rows - 1):
            for j in range(0, filled_cols - 1):
                if j >= int(free_spaces / 2) and j <= int(free_spaces / 2 + cols - 1):
                    filled[i, j] = img[i, int(j - free_spaces / 2)]
                else:
                    filled[i, j] = 0

    if cols > rows:
        free_spaces = NEW_SIZE - rows
        for i in range(0, filled_rows - 1):
            for j in range(0, filled_cols - 1):
                if i >= int(free_spaces / 2) and i <= int(free_spaces / 2 + rows - 1):
                    filled[i, j] = img[int(i - free_spaces / 2), j]
                else:
                    filled[i, j] = 0
    return filled
# Functions --------------------------------------- END

if len(sys.argv) < 2:
    print("No path to image set defined! Exiting...")
    exit()

# Algorithm --------------------------------------- START
location = str(sys.argv[1]) #'C:/Users/easmsma/Desktop/Diplomski/constellation-recognition/constellation-recognition/targets/lyra/positive/'
output = location + 'resized/'

if not os.path.exists(output) != False:
    print("Creating directory: " + output)
    os.mkdir(output)

counter = 0

for file in os.listdir(location):
    if file.endswith(".jpg"):
        print(file)
        # READ IMAGE (RGB)
        img = cv2.imread(location + file)


        rows, cols, _ = img.shape
        resized = 0
        filled = 0

        if rows > cols:
            rowsRatio = float(NEW_SIZE / rows)
            newCols = int(rowsRatio * cols)
            resized = resize_image(img, NEW_SIZE, newCols)
            filled = fill(resized)
        else:
            colsRatio = float(NEW_SIZE / cols)
            newRows = int(colsRatio * rows)
            resized = resize_image(img, newRows, NEW_SIZE)
            filled = fill(resized)

        new_file_name = str(output + os.path.splitext(file)[0] + "_resized.jpg");
        cv2.imwrite(new_file_name, filled)
        cv2.waitKey(0)
        counter = counter + 1

print("Total images resized: " + str(counter)),
# Algorithm --------------------------------------- END