#!/usr/bin/env python
""" Constellano Samples HAAR Preparation
Python script for samples preparation for the HAAR create samples process.

Command format:
    py PrepareSamples.py --pos <positives_dir> --neg <negatives_dir> -num <number_of_new_positive_samples_to_be_created> -maxxangle <max_x_rotation_angle> -maxyangle <max_y_rotation_angle> -maxzangle <max_z_rotation_angle>
"""

import os
import sys
import getopt
import shutil
import subprocess
from datetime import datetime

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.0.0"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = """Error! The command should be: py PrepareSamples.py --p <positives_dir> --n <negatives_dir> -num <number_of_new_positive_samples_to_be_created>
                                -maxxangle <max_x_rotation_angle> -maxyangle <max_y_rotation_angle> -maxzangle <max_z_rotation_angle>"""

def main(argv):
    positives_dir = ''
    negatives_dir = ''
    number_of_samples = ''
    maxxangle = ''
    maxyangle = ''
    maxzangle = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["pos=", "neg=", "num=", "maxxangle=", "maxyangle=", "maxzangle="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--pos"):
            positives_dir = arg
        elif opt in ("--neg"):
            negatives_dir = arg
        elif opt in ("--num"):
            number_of_samples = arg
        elif opt in ("--maxxangle"):
            maxxangle = arg
        elif opt in ("--maxyangle"):
            maxyangle = arg
        elif opt in ("--maxzangle"):
            maxzangle = arg

    start_time = datetime.now()

    # STEP 1 - Creating positives.txt and negatives.txt file
    if not os.path.exists(positives_dir) != False:
        print("[ERROR]\tPositive images " + positives_dir + " directory does not exist!")
        print("Please create it an re-run the script!")
        exit()
    if not os.path.exists(negatives_dir) != False:
        print("[ERROR]\tNegative images " + negatives_dir + " directory does not exist!")
        print("Please create it an re-run the script!")
        exit()

    subprocess.check_output("find " + positives_dir + " -iname '*.jpg' > positives.txt", shell=True)
    subprocess.check_output("find " + negatives_dir + " -iname '*.jpg' > negatives.txt", shell=True)
    print("[INFO]\tCreating positives.txt and negatives.txt.")

    # STEP 2 - Creating samples images for each positive file
    samples_dir =  "samples/"
    if not os.path.exists(samples_dir) != False:
        print("[INFO]\tCreating directory: " + samples_dir)
        os.mkdir(samples_dir)

    crutial_counter = 0
    for file in os.listdir(positives_dir):
        if file.endswith(".jpg"):
            current_samples_dir = str(samples_dir) + "samples_" + str(crutial_counter) + "/"
            current_samples_list = str(samples_dir) + "samples_" + str(crutial_counter) + "/samples_" + str(crutial_counter) + ".txt"
            if not os.path.exists(current_samples_dir) != False:
                print("[INFO]\tCreating directory: " + current_samples_dir)
                os.mkdir(current_samples_dir)
            command = "opencv_createsamples -img " + str(positives_dir + file) + " -bg negatives.txt " + "-info " + str(current_samples_list) + \
                            " -pngoutput " + str(current_samples_dir) + " -maxxangle " + str(maxxangle) + " -maxyangle " + str(maxyangle) + \
                            " -maxzangle " + str(maxzangle) + " -num " + \
                str(number_of_samples)
            response = subprocess.check_output(command, shell=True)
            print("[INFO]\tCreating samples from: " + file + " in: " + str(samples_dir) + "samples_" + str(crutial_counter))
            print(response)
            crutial_counter = crutial_counter + 1

    # STEP 3 - Moving all samples into a single directory and list
    final_samples_dir = "final_samples/"
    if not os.path.exists(final_samples_dir) != False:
        print("[INFO]\tCreating directory: " + final_samples_dir)
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

    print("[INFO]\tRemoving directory: " + samples_dir + "\tas it is of no use.")
    command = "rm -rf " + samples_dir   
    subprocess.check_output(command, shell=True)

    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")

if __name__ == "__main__":
    main(sys.argv[1:])