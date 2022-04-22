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

    crutial_counter = 0
    for file in os.listdir(positives_dir):
        if file.endswith(".jpg"):
            current_samples_dir = str(samples_dir) + \
                "samples_" + str(crutial_counter) + "/"
            current_samples_list = str(samples_dir) + \
                "samples_" + str(crutial_counter) + "/samples_" + str(crutial_counter) + ".txt"
            if not os.path.exists(current_samples_dir) != False:
                print("Creating directory: " + current_samples_dir)
                os.mkdir(current_samples_dir)
            command = "opencv_createsamples -img " + str(positives_dir + file) + " -bg negatives.txt " + "-info " + str(current_samples_list) + " -pngoutput " + str(current_samples_dir) + \
                " -maxxangle " + str(maxxangle) + " -maxyangle " + str(maxyangle) + \
                " -maxzangle " + str(maxzangle) + " -num " + \
                str(number_of_samples)
            response = subprocess.check_output(command, shell=True)
            print("Creating samples from: " + file + " in: " +
                  str(samples_dir) + "/samples_" + str(crutial_counter))
            crutial_counter = crutial_counter + 1

    # STEP 3 - Moving all samples into a single directory and list
    final_samples_dir = "final_samples/"
    if not os.path.exists(final_samples_dir) != False:
        print("Creating directory: " + final_samples_dir)
        os.mkdir(final_samples_dir)

    new_samples_list = []
    counter = 0
    for directory in os.listdir(samples_dir):
        current_samples_dir = str(samples_dir) + \
            "samples_" + str(counter) + "/"
        current_samples_list = []
        # Read list from samples_X.txt
        list_file = str(current_samples_dir) + \
            "samples_" + str(counter) + ".txt"
        with open(list_file, 'r') as listfile:
            for line in listfile.readlines():
                current_samples_list.append(line)

        inner_counter = 0
        for file in os.listdir(current_samples_dir):
            if file.endswith(".jpg"):
                source_file = str(current_samples_dir) + str(file)
                destination_file = str(final_samples_dir) + "final_sample_" + \
                    str(counter) + "_" + str(inner_counter) + ".jpg"
                shutil.copy(source_file, destination_file)

                for item in current_samples_list:
                    if file in item:
                        new_samples_list.append("final_sample_" + str(counter) + "_" + str(
                            inner_counter) + ".jpg" + str(item.split(".jpg")[1]))
            inner_counter = inner_counter + 1
        counter = counter + 1

    new_list_file_path = str(final_samples_dir) + "final_samples.txt"
    listfile = open(new_list_file_path, 'w')
    for line in new_samples_list:
        listfile.write(line)
    listfile.close()

    # STEP 4 - Generating positives.vec from final_samples
    #print("Generating positives.vec")
    #command = "opencv_createsamples -info final_samples/final_samples.txt -num " + \
    #    str(int(number_of_samples) * int(crutial_counter)) + " -w " + str(width) + \
    #    " -h " + str(height) + " + -vec positives.vec"
    #subprocess.check_output(command, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
