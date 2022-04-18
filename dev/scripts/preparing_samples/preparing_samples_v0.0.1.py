import os
import sys
import getopt
import struct
import subprocess

import glob
import argparse
import traceback


'''
The execution of this script will look like this:


py preparing_samples_v0.0.1.py -n neg/ -p pos/


'''

COMMAND_FORMAT = "Error! The command should be: preparing_samples.py -p <positives_dir> -n <negatives_dir> -num <number_of_new_positive_samples_to_be_created> -bgcolor <background_color> -bgthresh <background_color_threshold> -maxxangle <max_x_rotation_angle> -maxyangle <max_y_rotation_angle> -maxzangle <max_z_rotation_angle> -maxidev <max_intensity_deviation> -width <images_width> -height <images_height>"


def exception_response(e):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in lines:
        print(line)


def main(argv):
    positives_dir = ''
    negatives_dir = ''
    number_of_samples = ''
    bgcolor = ''
    bgthresh = ''
    maxxangle = ''
    maxyangle = ''
    maxzangle = ''
    maxidev = ''
    width = ''
    height = ''

    try:
        opts, args = getopt.getopt(argv, "hp:n:", [
            "num=", "bgcolor=", "bgthresh=", "maxxangle=",
            "maxyangle=", "maxzangle=", "maxidev=", "width=", "height="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("-p"):
            positives_dir = arg
        elif opt in ("-n"):
            negatives_dir = arg
        elif opt in ("--num"):
            number_of_samples = arg
        elif opt in ("--bgcolor"):
            bgcolor = arg
        elif opt in ("--bgthresh"):
            bgthresh = arg
        elif opt in ("--maxxangle"):
            maxxangle = arg
        elif opt in ("--maxyangle"):
            maxyangle = arg
        elif opt in ("--maxzangle"):
            maxzangle = arg
        elif opt in ("--maxidev"):
            maxidev = arg
        elif opt in ("--width"):
            width = arg
        elif opt in ("--height"):
            height = arg

    # STEP 1 - Creating positives.txt and negatives.txt file
    if not os.path.exists(positives_dir) != False:
        print("Positive images " + positives_dir + " directory does not exist!")
        print("Please create it an re-run the script!")
        exit()
    if not os.path.exists(negatives_dir) != False:
        print("Negative images " + negatives_dir + " directory does not exist!")
        print("Please create it an re-run the script!")
        exit()

    subprocess.check_output(
        "find " + positives_dir + " -iname '*.jpg' > positives.txt", shell=True)
    subprocess.check_output(
        "find " + negatives_dir + " -iname '*.jpg' > negatives.txt", shell=True)
    print("Creating positives.txt and negatives.txt DONE!")

    # STEP 2 - Creating samples images for each positive file
    samples_dir = positives_dir + "samples/"
    if not os.path.exists(samples_dir) != False:
        print("Creating directory: " + samples_dir)
        os.mkdir(samples_dir)

    counter = 0
    for file in os.listdir(positives_dir):
        if file.endswith(".jpg"):
            command = "opencv_createsamples -img " + str(samples_dir + file) + " -bg negatives.txt " + "-info " + str(samples_dir) + "/samples_" + str(counter) + ".txt -pngoutput " + str(samples_dir) + "/samples_" + str(counter) + \
                " -maxxangle " + str(maxxangle) + " -maxyangle " + str(maxyangle) + " -maxzangle " + str(maxzangle) + " -num " + str(number_of_samples) + \
                " -maxidev " + str(maxidev) + " -w " + \
                str(width) + " -h " + str(height)
            subprocess.check_output(command, shell=True)
            print("Creating samples from: " + file + " in: " +
                  str(samples_dir) + "/samples_" + str(counter))
            counter = counter + 1

    # STEP 3 - Creating a list of all samples .vec files
    subprocess.check_output(
        "find " + samples_dir + " -iname '*.vec' > samples.txt", shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])


'''
command = "opencv_createsamples -vec " + str(samples_dir) + "samples_" + str(counter) + ".vec -img " + positives_dir + file + " -bg negatives.txt -num " + str(number_of_samples) + " -bgcolor " + str(bgcolor) + " -bgthresh " + \
                str(bgthresh) + " -maxxangle " + str(maxxangle) + " -maxyangle " + str(maxyangle) + " -maxzangle " + \
                str(maxzangle) + " -maxidev " + str(maxidev) + \
                " -w " + str(width) + " -h " + str(height)
'''
