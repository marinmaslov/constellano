import os
import sys
import getopt
import shutil
import traceback
import subprocess


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


    # STEP 4 - Generating positives.vec from final_samples
    print("Generating positives.vec")
    command = "opencv_createsamples -info " + str(samples) + " -num " + \
        str(number_of_samples) + " -w " + str(width) + \
        " -h " + str(height) + " + -vec positives.vec"
    subprocess.check_output(command, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
