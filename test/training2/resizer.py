import sys
import os
import cv2
import numpy as np


import os
import sys
import getopt
import shutil
import traceback
import subprocess


COMMAND_FORMAT = "Error! The command should be: resizer.py -d <images_dir> -s <image_size>"


def resize_image(img, newRows, newCols):
    return cv2.resize(img, (int(newCols), int(newRows)), interpolation=cv2.INTER_AREA)


def fill(img, size):
    filled = np.zeros((int(size), int(size), 3), np.uint8)

    rows, cols, _ = img.shape
    filled_rows, filled_cols, _ = filled.shape

    if rows > cols:
        free_spaces = int(size) - cols
        for i in range(0, filled_rows - 1):
            for j in range(0, filled_cols - 1):
                if j >= int(free_spaces / 2) and j <= int(free_spaces / 2 + cols - 1):
                    filled[i, j] = img[i, int(j - free_spaces / 2)]
                else:
                    filled[i, j] = 0

    if cols > rows:
        free_spaces = int(size) - rows
        for i in range(0, filled_rows - 1):
            for j in range(0, filled_cols - 1):
                if i >= int(free_spaces / 2) and i <= int(free_spaces / 2 + rows - 1):
                    filled[i, j] = img[int(i - free_spaces / 2), j]
                else:
                    filled[i, j] = 0
    return filled


def main(argv):
    images_dir = ''
    image_size = ''

    try:
        opts, args = getopt.getopt(argv, "hd:s:")
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("-d"):
            images_dir = arg
        elif opt in ("-s"):
            image_size = arg

    # Algorithm --------------------------------------- START
    # 'C:/Users/easmsma/Desktop/Diplomski/constellation-recognition/constellation-recognition/targets/lyra/positive/'
    location = str(images_dir)
    output = location + 'resized/'

    if not os.path.exists(output) != False:
        print("Creating directory: " + output)
        os.mkdir(output)

    counter = 0

    for file in os.listdir(location):
        print(file)
        if file.endswith(".jpg"):
            print(file)
            # READ IMAGE (RGB)
            img = cv2.imread(location + file)

            rows, cols, _ = img.shape
            resized = 0
            filled = 0

            if rows > cols:
                rowsRatio = float(int(image_size) / rows)
                newCols = int(rowsRatio * cols)
                resized = resize_image(img, int(image_size), newCols)
                filled = fill(resized, int(image_size))
            else:
                colsRatio = float(int(image_size) / cols)
                newRows = int(colsRatio * rows)
                resized = resize_image(img, newRows, int(image_size))
                filled = fill(resized, int(image_size))

            new_file_name = str(
                output + os.path.splitext(file)[0] + "_resized.jpg")
            cv2.imwrite(new_file_name, filled)
            cv2.waitKey(0)
            counter = counter + 1

    print("Total images resized: " + str(counter)),
    # Algorithm --------------------------------------- END


if __name__ == "__main__":
    main(sys.argv[1:])
