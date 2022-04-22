#!/usr/bin/env python
""" Constellano HAAR Create Samples
Python script for HAAR sample creation.

Command format:
    py create_samples.py -d <samples.txt> --num <number_of_samples_to_use> --width <width_for_haar_training> --height <height_for_haar_training>

Command example:
    py create_samples.py -d final_dir/samples.txt --num 1000 -w 24 -h 24
"""

import sys
import getopt
import traceback
import subprocess

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "1.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py create_samples.py -d <samples.txt> --num <number_of_samples_to_use> --width <width_for_haar_training> --height <height_for_haar_training>"


def exception_response(e):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in lines:
        print(line)


def main(argv):
    samples = ''
    number_of_samples = ''
    width = ''
    height = ''

    try:
        opts, args = getopt.getopt(argv, "hd:", [
            "num=", "width=", "height="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("-d"):
            samples = arg
        elif opt in ("--num"):
            number_of_samples = arg
        elif opt in ("--width"):
            width = arg
        elif opt in ("--height"):
            height = arg


    # Generating positives.vec from final_samples
    print("Generating positives.vec")
    command = "opencv_createsamples -info " + str(samples) + " -num " + \
        str(number_of_samples) + " -w " + str(width) + \
        " -h " + str(height) + " + -vec positives.vec"
    subprocess.check_output(command, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])