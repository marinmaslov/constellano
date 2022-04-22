#!/usr/bin/env python
""" Constellano HAAR Training
Python script for HAAR training.

Command format:
    py haar_training.py --data <positives_dir> --vec <samples.vec> --bg <negatives_dir> --numPos <number_of_positives> --numNeg <number_of_negatives>
                            --numStages <number_of_stages> --width <width_for_haar_training> --height <height_for_haar_training> --mode ALL --bt DAB
                            --minHitRate 0.995 --maxFalseAlarmRate 0.5 --maxWeakCount 100 --maxDepth 1 --precalcValBufSize 1024 --precalcIdxBufSize 1024

Command example:
    py haar_training.py --data data/ --vec samples.vec --bg negs/ --numPos 1000 --numNeg 1500 --numStages 20 --width 24 --height 24 --mode ALL --bt DAB
                            --minHitRate 0.995 --maxFalseAlarmRate 0.5 --maxWeakCount 100 --maxDepth 1 --precalcValBufSize 1024 --precalcIdxBufSize 1024
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

COMMAND_FORMAT = "Error! The command should be: py haar_training.py --data data/ --vec samples.vec --bg negs/ --numPos 1000 --numNeg 1500 --numStages 20 --width 24 --height 24 --mode ALL --bt DAB --minHitRate 0.995 --maxFalseAlarmRate 0.5 --maxWeakCount 100 --maxDepth 1 --precalcValBufSize 1024 --precalcIdxBufSize 1024"


def exception_response(e):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in lines:
        print(line)

def main(argv):
    data = ''
    vec = ''
    neg = ''
    numPos = ''
    numNeg = ''
    numStages = ''
    width = ''
    height = ''
    mode = ''
    bt = ''
    minHitRate = ''
    maxFalseAlarmRate = ''
    maxWeakCount = ''
    maxDepth = ''
    precalcValBufSize = ''
    precalcIdxBufSize = ''

    try:
        opts, args = getopt.getopt(argv, "h", [
            "data=", "vec=", "neg=", "numPos=", "numNeg=", "numStages=", "width=", "height=", "mode=", "bt=",
            "minHitRate=", "maxFalseAlarmRate=", "maxWeakCount=", "maxDepth=", "precalcValBufSize=", "precalcIdxBufSize="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--data"):
            data = arg
        elif opt in ("--vec"):
            vec = arg
        elif opt in ("--neg"):
            neg = arg
        elif opt in ("--numPos"):
            numPos = arg
        elif opt in ("--numNeg"):
            numNeg = arg
        elif opt in ("--numStages"):
            numStages = arg
        elif opt in ("--width"):
            width = arg
        elif opt in ("--height"):
            height = arg
        elif opt in ("--mode"):
            mode = arg
        elif opt in ("--bt"):
            bt = arg
        elif opt in ("--minHitRate"):
            minHitRate = arg
        elif opt in ("--maxFalseAlarmRate"):
            maxFalseAlarmRate = arg
        elif opt in ("--maxWeakCount"):
            maxWeakCount = arg
        elif opt in ("--maxDepth"):
            maxDepth = arg
        elif opt in ("--precalcValBufSize"):
            precalcValBufSize = arg
        elif opt in ("--precalcIdxBufSize"):
            precalcIdxBufSize = arg


    # Generating positives.vec from final_samples
    print("Running HAAR training")
    command = "opencv_traincascade --data " + str(data) + " --vec " + str(vec) + " --bg " + str(neg) + " --numPos " + str(numPos) + " --numNeg " + str(numNeg) +\
        + " --numStages " + str(numStages) + " --width " + str(width) + " --height " + str(height) + " --mode " + str(mode) + " --bt " + str(bt) + " --minHitRate " + str(minHitRate) +\
        + " --maxFalseAlarmRate " + str(maxFalseAlarmRate) + " --maxWeakCount " + str(maxWeakCount) + " --maxDepth " + str(maxDepth) + " --precalcValBufSize " + str(precalcValBufSize) + " --precalcIdxBufSize " + str(precalcIdxBufSize)
    subprocess.check_output(command, shell=True)

if __name__ == "__main__":
    main(sys.argv[1:])