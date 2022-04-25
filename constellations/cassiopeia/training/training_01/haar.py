#!/usr/bin/env python
""" Constellano Star Recognition
Python script for finding biggest and brightest stars in images and overlaying a target over them.

Command format:
    py resizer.py -d <image_dir> -s <wanted_image_size>

Command example:
    py resizer.py -d img/ -s 500
"""

import sys
import os
import cv2
import numpy as np
import getopt

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "1.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

COMMAND_FORMAT = "Error! The command should be: py resizer.py -d <images_dir> -s <image_size>"
d1x, d1y, d1ra, d1dec, d2x, d2y, d2ra, d2dec = (0,)*8

def main(argv):
    cascade = ''
    image = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["cascade=", "image="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--cascade"):
            cascade = arg
        elif opt in ("--image"):
            image = arg

    # HAAR --------------------------------------- START
    cascade = cv2.CascadeClassifier(cascade)

    img = cv2.imread(image)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    constellations, rejectLevels, levelWeights = cascade.detectMultiScale3(
    	img,
    	scaleFactor = 1.05,
        minNeighbors = int(3),
        flags = 0,
        #minSize = (300, 300),
        #maxSize = (400, 400),
        outputRejectLevels = True
	)

    i = 0
    highweight = 0
    big_w = 0
    weighted = 0

	if(len(constellations) > 0):
        for (x,y,w,h) in constellations:
            #This if statement will find the detection with the largest bounding box.
            if w > big_w:
                highweight = levelWeights[i]
                weighted = float(highweight)*float(sf)
                x1 = x
                y1 = y
                w1 = w
                h1 = h

        if (weighted > 4) and (weighted < 6):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,cascade,(x1,y1-16), font,0.9,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(img,str(weighted),(x1,y1+h1+25), font,0.7,(0,0,255),2,cv2.LINE_AA)
            cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
            cenpixx = int(x1 + 0.5 * w1)
            cenpixy = int(y1 + 0.5 * h1)
            cv2.putText(img,str(cenpixx)+', '+str(cenpixy),(x1,y1-45), font,0.9,(0,0,255),2,cv2.LINE_AA)
            shrunk_img = cv2.resize(img, (1344, 756))
            cv2.imshow("Star Pattern Detections",shrunk_img)
            print('Cascade number '+cascade+' DETECTS')
            print(weighted)
            print()

            #Pulls in the global variables for the pixel and world coordinates of the detections.
            global d1x, d1y, d1ra, d1dec
            global d2x, d2y, d2ra, d2dec

            #The following statements assign the parameters of two successful detection to those variables.
            if (d1x == 0):
                d1x = cenpixx
                d1y = cenpixy
                d1ra = ra
                d1dec = dec
            else:
                d2x = cenpixx
                d2y = cenpixy
                d2ra = ra
                d2dec = dec
        else:
            print('Cascade number '+cascade+' POOR DETECTION')
            print(weighted)
            print()

    else:
        print('Cascade number '+cascade+' NO DETECTION')
        print()





    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow('img',img)
    # HAAR --------------------------------------- END

if __name__ == "__main__":
    main(sys.argv[1:])

