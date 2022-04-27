#!/usr/bin/env python
""" Constellano Samples HAAR Preparation
Python script for samples preparation for the HAAR create samples process.

Command format:
    py preparing_samples.py -p <positives_dir> -n <negatives_dir> --num <number_of_new_positive_samples_to_be_created> --maxxangle <max_x_rotation_angle> --maxyangle <max_y_rotation_angle> --maxzangle <max_z_rotation_angle>

Command example:
    py preparing_samples.py -p pos/ -n neg/ --num 1000 --maxxangle 1.1 --maxyangle 1.1 --maxzangle 0.5
"""

import os
import sys
import getopt
import shutil
import traceback
import subprocess
import time

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "1.0.2"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py preparing_samples.py -p <positives_dir> -n <negatives_dir> -num <number_of_new_positive_samples_to_be_created> -maxxangle <max_x_rotation_angle> -maxyangle <max_y_rotation_angle> -maxzangle <max_z_rotation_angle>"

def exception_response(e):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in lines:
        print(line)


def main(argv):
    positives_dir = ''
    negatives_dir = ''
    number_of_samples = ''
    maxxangle = ''
    maxyangle = ''
    maxzangle = ''

    try:
        opts, args = getopt.getopt(argv, "hp:n:", [
            "num=", "maxxangle=",
            "maxyangle=", "maxzangle="])
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
        elif opt in ("--maxxangle"):
            maxxangle = arg
        elif opt in ("--maxyangle"):
            maxyangle = arg
        elif opt in ("--maxzangle"):
            maxzangle = arg

    # STEP 0 - Get current file path
    #current_file_name = __file__.replace("\\", "/").split("/")[-1]
    #current_file_parent_directory = __file__.replace(current_file_name, "").replace("\\", "/")

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
    samples_dir =  "samples/"
    if not os.path.exists(samples_dir) != False:
        print("Creating directory: " + samples_dir)
        os.mkdir(samples_dir)

    crutial_counter = 0
    for file in os.listdir(positives_dir):
        if file.endswith(".jpg"):
            current_samples_dir = str(samples_dir) + "samples_" + str(crutial_counter) + "/"
            current_samples_list = str(samples_dir) + "samples_" + str(crutial_counter) + "/samples_" + str(crutial_counter) + ".txt"
            if not os.path.exists(current_samples_dir) != False:
                print("Creating directory: " + current_samples_dir)
                os.mkdir(current_samples_dir)
            command = "opencv_createsamples -img " + str(positives_dir + file) + " -bg negatives.txt " + "-info " + str(current_samples_list) + \
                            " -pngoutput " + str(current_samples_dir) + " -maxxangle " + str(maxxangle) + " -maxyangle " + str(maxyangle) + \
                            " -maxzangle " + str(maxzangle) + " -num " + \
                str(number_of_samples)
            response = subprocess.check_output(command, shell=True)
            print("[INFO]\tCreating samples from: " + file + " in: " + str(samples_dir) + "samples_" + str(crutial_counter))
            crutial_counter = crutial_counter + 1

    # STEP 3 - Moving all samples into a single directory and list
    final_samples_dir = "final_samples/"
    if not os.path.exists(final_samples_dir) != False:
        print("Creating directory: " + final_samples_dir)
        os.mkdir(final_samples_dir)

    new_samples_list = []
    counter = 0
    for directory in os.listdir(samples_dir):
        current_samples_dir = str(samples_dir) + "samples_" + str(counter) + "/"
        current_samples_list = []
        # Read list from samples_X.txt
        list_file = str(current_samples_dir) + "samples_" + str(counter) + ".txt"
        with open(list_file, 'r') as listfile:
            for line in listfile.readlines():
                current_samples_list.append(line)

        inner_counter = 0
        for file in os.listdir(current_samples_dir):
            if file.endswith(".jpg"):
                source_file = str(current_samples_dir) + str(file)
                destination_file = str(final_samples_dir) + "final_sample_" + str(counter) + "_" + str(inner_counter) + ".jpg"
                shutil.copy(source_file, destination_file)

                for item in current_samples_list:
                    if file in item:
                        new_samples_list.append("final_sample_" + str(counter) + "_" + str(inner_counter) + ".jpg" + str(item.split(".jpg")[1]))
            inner_counter = inner_counter + 1
        counter = counter + 1

    new_list_file_path = str(final_samples_dir) + "final_samples.txt"
    listfile = open(new_list_file_path, 'w')
    for line in new_samples_list:
        listfile.write(line)
    listfile.close()

    print("Removing directory: " + samples_dir + "\tas it is of no use.")
    command = "rm -rf " + samples_dir   
    subprocess.check_output(command, shell=True)

if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    print("\033[2;32;40m[INFO]\033[0;0m" + "\tTotal execution time: " + str((time.time() - start_time)) + " seconds.\033[0;0m")