#!/usr/bin/env python
""" Constellano: HAAR Training

Python script for the execution opencv_cascadetraining commands with custom parameters.

Command example:
    py resizer.py -d <images-directory> -s <image-size>
"""

import os
import sys
import getopt
import shutil
import traceback
import subprocess

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "1.0.0"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: preparing_samples.py -p <positives_vec_file> -n <negatives_list_file> -numPos <number_of_positives_to_use> -numNeg <number_of_negatives_to_use> -numStages <number_of_stages_to_run> -width <images_width> -height <images_height>"


def main(argv):
    positives_dir = ''
    negatives_dir = ''
    numpos = ''
    numneg = ''
    numstages = ''
    width = ''
    height = ''

    try:
        opts, args = getopt.getopt(argv, "hp:n:", [
            "numPos=", "numNeg=", "numStages=", "width=", "height="])
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
        elif opt in ("--numPos"):
            numpos = arg
        elif opt in ("--numNeg"):
            numneg = arg
        elif opt in ("--numStages"):
            numstages = arg
        elif opt in ("--width"):
            width = arg
        elif opt in ("--height"):
            height = arg

    # Running HAAR cascade training

    destination_data_dir = "data/"
    if not os.path.exists(destination_data_dir) != False:
        print("Creating directory: " + destination_data_dir)
        os.mkdir(destination_data_dir)

    print("Running HAAR cascade training...")
    command = "opencv_traincascade -data " + str(destination_data_dir) + " -vec " + str(positives_dir) + " -bg " + str(negatives_dir) + " -numPos " + str(
        numpos) + " -numNeg " + str(numneg) + " -numStages " + str(numstages) + " -w " + str(width) + " -h " + str(height)
    subprocess.check_output(command, shell=True)
    print("HAAR cascade training finished!")


if __name__ == "__main__":
    main(sys.argv[1:])
