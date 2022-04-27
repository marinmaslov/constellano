# 🤖 HAAR training branch

## ⚙️ Data preparation procedure

### 🚀 STEP 1: Run the Stellarium scripts
Copy the `grep_negatives.ssc` and `grep_positives.ssc` scripts to Stellarium scripts directory.

Find the bounding coordinates of your prefered targed (on the date: 2000-01-01 00:00:00 UTC+02:00), and put them into the scripts. Also, change the `DESTINATION_PATH` in the scripts to your desired (and existing) images storing directory.

Run each script, but before running hide every possible label (for stars, planets, etc.) in Stellarium!

Now you have your positive and negative data.

Move them to your training directory, along with the following scripts, so that the directory structure will look like this:

    .
    └── training_n
            |── positives/
            |       |── img_000.jpg
            |       |── ...
            |       └── img_nnn.jpg
            |── negatives/
            |       |── img_000.jpg
            |       |── ...
            |       └── img_nnn.jpg
            |── star_detector.py
            |── resizer.py
            |── rotator.py
            └── preparing_samples.py

### 🚀 STEP 2: Run the Star detector script on positive images

Hit the command:
```bash
py star_detector.py --images <positives_directory_(e.g. positives/)> --percision <percision_rate_(e.g. 0.18)> --log <log_level_(e.g. INFO)>
```

This script should only be run on the positive images.

After the execution finishes, a output directory containing the images with detected stars will appear within the positives directory.

Crop all files so only the targeted constellation is shown.

### 🚀 STEP 3: Run the Resizer script
Before using images for the HAAR cascade, we should resize them as the proccess of genereting the vector file and the cascade file will use a lot of proccessing power. 

Hit the command:
```bash
py resizer.py --images <images_directory_(e.g. positives/)> --size <size_of_ste_output_images_1:1_aspect_ratio> --grayscale <0_if_you_want_bw_images> --log <log_level_(e.g. INFO)>
```

Run this script on your negative and positive (where the stars have been detected) images.

Now you the resized negative and positive (with detected stars) images.

### 🚀 STEP 4: Run the PRotator script
Constellations can be rotated in any direction depending on the date, and location, on which you thake photographs of them. For HAAR to be very accurate, we'll also prepare rotated positive samples. The opencv_createsamples command also provides parameters for the max angles on x, y and z axis, but when rotating images on z axis it cuts out part of the image. Therefore, we'll rotate the images on our own.

Run this command only on positive images with detected stars.

Hit the command:
```bash
py rotator.py --images <images_directory_(e.g. positives/)> --maxangle <e.g. 360> --log <log_level_(e.g. INFO)>
```

### 🚀 STEP 5: Run the Preparing samples script
This script will generate new positives overlayed over the negatives, which will be used for generating the vector file for cascade training.

Hit the command:
```bash
py resizer.py -p <rotated_positives_directory> -n <negatives_directory> --num <number_of_new_positives_to_be_created_for_each_existing_positive> --maxxangle 0.0 --maxyangle 0.0 --maxzangle 0.0
```

After executin this script, you will have your final_example positive images, and three lists that will be used for the generation of the .vec file.

### 🚀 STEP 6: Generate the vector file
Execute the following command:
```bash
opencv_createsamples -info final_samples/final_samples.txt -num 10000 -w 24 -h 24 -vec positives.vec -maxxangle 0.0 -maxyangle 0.0 -maxzangle 0.0
```

As output you should get the .vec file.

### 🚀 STEP 7: Run the HAAR training
Execute the following command:
```bash
opencv_traincascade -data p/ -vec positives.vec -bg negatives.txt -numPos 1000 -numNeg 1200 -numStages 20 -width 24 -height 24 -mode ALL -bt DAB -minHitRate 0.995 -maxFalseAlarmRate 0.5 -maxWeakCount 100 -maxDepth 1 -precalcValBufSize 1024 --precalcIdxBufSize 1024
```

As output you should get the .xml cascade file (within the positive images directory).

## Sample creation procedure

## HAAR training procedure

## HAAR training results

🎯 *Training target:* Lyra Constellation

📝 *Target description:* Lyra consists of 6 bigger stars and multiple other smaller stars. The cascade will be trained to recognize those 6 brightest stars.

📋 *Training overall description:* A few training attempt will be made in hope that a `cascade.xml` will be generated that will be able to recognize the marked pattern of Lyra. Each training attempt will be described, it's training parameters will be listed, and finally the results will be shown (and later compared altogether). If the results of no attempt will be able to recognize the constellation, another set of training attempts will be made on a set of data where the stars are not just marked, but also connected.

### 🏋️ Training attepmt #1 
#### 🗃️ Input data

#### 🦾 Training parameters

#### 📈 Results

### 🏋️ Training attepmt #2
#### 🗃️ Input data

#### 🦾 Training parameters

#### 📈 Results

### 🏋️ Training attepmt #3
#### 🗃️ Input data

#### 🦾 Training parameters

#### 📈 Results

### 🏋️ Training attepmt #4
#### 🗃️ Input data

#### 🦾 Training parameters

#### 📈 Results