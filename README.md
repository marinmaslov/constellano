# 🌟 Constellano: Star recognition algorithm 🌌 💻

An app written in Python (Flask) that enables you to recognize 👀 constellations on a static picture, using HAAR cascade 🤖.

## 1. Prerequisite ⚙️
To run this app you'll need to install `python 3.10.x`.

## 2. How to run? 🚀
Clone the repo using the following command:
```bash
git clone git@github.com:marinmaslov/constellano.git
```
Position yourself into the constellano directory:
```bash
cd constellano
```
Create a virtual environment:
```bash
python -m venv venv
```
Install all required dependencies:
```bash
pip install -r requirements.txt
```
Activate it:
```bash
source venv/bin/activate
```
Install all required modules (make sure you're is the same directory where the requirements.txt file is):
```bash
pip install -r requirements.txt
```
Run the app with the following command (again make sure you're in the same directory as the app.py file):
```bash
flask run
```

## 3. What's inside? 🧐
A quick look at the apps files and directories.

    .
    ├── cascades
    |       |── pretrained_001.xml
    |       |── ...
    |       └── pretrained_NNN.xml
    ├── data
    |       |── negatives
    |       └── positives
    ├── scripts
    |       └── all scripts explained in section 4.
    ├── testing
    |       └── mostly testing images
    ├── .gitignore
    ├── Procfile
    ├── README.md
    ├── app.py
    ├── requirements.txt
    └── runtime.txt

## 4. Documentation 📚
### 4.1. Introduction
In this section an overview on how to use all the scripts to prepare data and train HAAR to detect the input object will be explained.
### 4.2. Data preparation
#### 4.2.1. Stellarium scripts
Fisrtly, we need to collect our data. Instead of shooitng the sky and waisting hundreds of hours on it, we'll use "laboratory" data which we'll fetch from an open-source software called Stellarium. A detailed approach on how to import and run scripts in Stellraium can be found [here!](https://stellarium.org/doc/0.20/scripting.html)

We have two scripts: one for fetching **positive** images (the ones containing the object we want to be able to detect) and one for fetching **negative** images (all other parts of the sky without that object), which can be found in `scripts/`

##### Script for fetching positives
Before running this script, a few constants should be changed.
- `DESTINATION_PATH` represents the system path where the images will be stored
- `RA_TARGET_START` starting RA coordinate (in decimal degrees) for the object of interest
- `RA_TARGET_END` ending RA coordinate (in decimal degrees) for the object of interest
- `DEC_TARGET_START` starting DEC coordinate (in decimal degrees) for the object of interest
- `DEC_TARGET_END` ending DEC coordinate (in decimal degrees) for the object of interest

After running this script you'll end up with a few (max. 30) images that contain the object of interest.

##### Script for fetching negatives
This script will take screen shoots of the whole sky, just change the following constant.
- `DESTINATION_PATH` represents the system path where the images will be stored

After running this script you'll end up with more then 10k images of the sky.

#### 4.2.2. Star detection
The first step is to apply star masks onto the brightest stars in the positive images. To apply the star masks run the script `scripts/StarDetector.py` as follows:

```bash
py scripts/StarDetector.py --images PATH_TO_IMAGES_DIR --masksize MASK_SIZE_PERCENTAGE --outputname OUTPUT_NAME --percision PERCISION_PERCENTAGE
```

The parameters are:
- `PATH_TO_IMAGES_DIR` relative path to the directory containing the positive images
- `MASK_SIZE_PERCENTAGE` mask size percentage in realtion to the input image's width (e.g.  `0.06`  means  `6%` × input image's width)
- `OUTPUT_NAME` name that will be given to the output files
- `PERCISION_PERCENTAGE` percentage of the brightest star that will be used as the a thrashold value. To all the stars having an area size smaller then the threshold value won't be covered by the mask. Usage is similar to `MASK_SIZE_PERCENTAGE` (e.g. `0.18` means `18%`)

#### 4.2.3. Image cropping
After successfully appyling masks onto stars crop the image so only the object of interest will be visible.

#### 4.2.4. Resizer
The next step is to resize all images to some desired dimensions (e.g. 500×500px).

Run the script as follows:
```bash
py scripts/Resizer.py --images PATH_TO_IMAGES_DIR --size SIZE --grayscale VALUE
```

The parameters are:
- `PATH_TO_IMAGES_DIR` relative path to the directory containing the cropped positive images
- `SIZE` size in pixels (size × size) for the output images
- `VALUE` if images need to be converted to grayscale set `0`, if not set to any other number

#### 4.2.4. Generating more positive samples
This step will generate many new positive images (with distorsions) that will be used to train the cascade. The new images are generated by applying the exisitng positive images onto the negative images but with distorisions.

As this script uses the openCV library as a system module, please install openCV as a system module (perhaps the best is to run it on Linux as it is the easiest to install openCV as a system module there).

Run the script as follows:
```bash
python scripts/PrepareSamples.py --pos POSITIVES_DIR --neg NEGATIVES_DIR --num NUMBER_OF_NEW --maxxangle MAX_X_ANGLE --maxyangle MAX_Y_ANGLE --maxzangle MAX_Z_ANGLE
```

The parameters are:
- `POSITIVES_DIR` relative path to the directory containing the positive images
- `NEGATIVES_DIR` relative path to the directory containing the negative images
- `NUMBER_OF_NEW` number of positives that will be genmerated for each existing positive
- `MAX_X_ANGLE` the max. angle on the x-axis the positives will be rotated while appyling them on the negatives (best opetion `0.0`)
- `MAX_Y_ANGLE` the max. angle on the y-axis the positives will be rotated while appyling them on the negatives (best opetion `0.0`)
- `MAX_Z_ANGLE` the max. angle on the z-axis the positives will be rotated while appyling them on the negatives (best opetion `0.0`)

After executing this script a directory named `final_samples` will be created containing all new positive images and the `final_samples.txt` file needed for the newx step. Also two files named `positives.txt` and `negatives.txt` will appear.

#### 4.2.5.  Generating the HAAR vector
To generate the input vector for the HAAR training, run the following openCV command:
```bash
opencv_createsamples -info SAMPLES_LIST -num NUMBER -w WIDTH -h HEIGHT -vec VECOTR -maxxangle MAX_X_ANGLE -maxyangle MAX_Y_ANGLE -maxzangle MAX_Z_ANGLE
```

The parameters are:
- `SAMPLES_LIST` list file (`.txt`) with all positive samples (`final_samples/final_samples.txt` from last step)
- `NUMBER` number of positive images that will be used for the creation of the vector file
- `WIDTH` width of the input images (last step used `24`)
- `HEIGHT` height of the input images (last step used `24`)
- `VECOTR` name of the output vector file (e.g. `positives.vec`)
- `MAX_X_ANGLE` the max. angle on the x-axis the positives have been rotated
- `MAX_Y_ANGLE` the max. angle on the y-axis the positives have been rotated
- `MAX_Z_ANGLE` the max. angle on the z-axis the positives have been rotated

### 4.3. HAAR training
After preparing all input files for the HAAR training process, hit the following command:
```bash
opencv_traincascade -data DATA -vec VECOTR -bg NEGATIVES_LIST -numPos NUMBER_POS -numNeg NUMBER_NEG -numStages NUMBER_STAGES -width WIDTH -height HEIGHT -mode ALL -bt DAB -minHitRate 0.995 -maxFalseAlarmRate 0.5 -maxWeakCount 100 -maxDepth 1 -precalcValBufSize 1024 -precalcIdxBufSize 1024
```

The parameters are:
- `DATA` directory with final samples from last steps
- `VECOTR` vector file from last step
- `NEGATIVES_LIST` negatives list from last step (`negatives.txt`)
- `NUMBER_POS` number of positive images that will be used to train the cascade
- `NUMBER_NEG` number of negative images that will be used to train the cascade
- `NUMBER_STAGES` number of stages HAAR will take to train the cascade (min. 1, max. 20)
- `WIDTH` width of the input images (last step used `24`)
- `HEIGHT` height of the input images (last step used `24`)

Other parameters should be kept as they are.

After the training finishes, an output file named `cascade.xml` will be created in the directory `final_samples/`.

### 4.4. Detecting objects using the generated HAAR cascade
The HAAR detection script can be run using the following command:
```bash
python scripts/HaarDetection.py --images IMAGES_DIR --masksizeMin MASK_SIZE_MIN --masksizeMax MASK_SIZE_MAX --outputname OUTPUT_NAME --percisionMin PERCISION_MIN --percisionMax PERCISION_MAX --cascade CASCADES_DIR --scale 1.01 --minNghb 2 --json JSON_FILE --plot 0 --streach 0
```

The parameters are:
- `IMAGES_DIR` path to dircetory containing the input images on which you want to detect objects
- `MASK_SIZE_MIN` minimum mask size in percentage as described in 4.2.2.
- `MASK_SIZE_MAX` maximum mask size in percentage as described in 4.2.2.
- `OUTPUT_NAME` name of the output directory and files
- `PERCISION_MIN` minimum percision in percentage as described in 4.2.2. (preferably `0.0`)
- `PERCISION_MAX` maximum percision in percentage as described in 4.2.2.
- `CASCADES_DIR` directory containing all the cascade files (the output files from step 4.3. should be placed into this directory and renamed to the desired name of the object question)
- `JSON_FILE` path to the json file

Other parameters should be as they are. If detection is unsucessful change the `streach` parameter to `1`, if you want to plot every image in every step of the detectio script change the `plot` parameter to `1`.



> "The real friends of the space voyager are the stars. Their friendly, familiar patterns are constant companions, unchanging, out there." - James Lovell, Apollo Astronaut

<p align="center">
  Python Script created by Marin Maslov @ <a href="https://www.fesb.unist.hr/">FESB (UNIST)</a>
</p>
