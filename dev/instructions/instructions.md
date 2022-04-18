# Constellano developer instructions






## Creating test samples

### Creating list files
#### Negatives
List files are the easiest to prepare. Just put all your negatives into a directory called `neg` and generate a file called `neg.txt` containing a list of all the negative images.

Use the following command to generate the list of negative images:

```bash
find neg/ -iname "*.jpg" > neg.txt
```

When the command execution is finished a `neg.txt` file should appear and it should contain all the names of the files in the `neg` directory.

The overall testing directory structure, for now should be:

```txt
.
├── neg
|	|── neg_img_001.jpg
|	|── ...
|	└── neg_img_NNN.jpg
└── neg.txt
```

#### Positives
The same way you created the `neg.txt` will be used to create `pos.txt`.

Put all your positive images in a folder called `pos`, and generate the `pos.txt` using the following command:

```bash
find pos/ -iname "*.jpg" > pos.txt
```

The overall testing directory structure, after creating the `neg.txt` and `pos.txt`, should be:

```txt
.
├── neg
|	|── neg_img_001.jpg
|	|── ...
|	└── neg_img_NNN.jpg
├── pos
|	|── pos_img_001.jpg
|	|── ...
|	└── pos_img_NNN.jpg
├── neg.txt
└── pos.txt
```

### Creating samples
We need positive and negative samples for the HAAR cascade training process. The negative samples are already in place as they are just the arbitrary images we put into the `neg` directory, and for which we already created the `neg.txt` file that contains a list of all the negatives.

What we do need to create are the positive samples. 

For creating positive samples, we'll use the `opencv_createsamples` command for which we need a couple of arguments as described in the opencv <a href="https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html#positive-samples">documentation</a>:

We need to define the output file:
`-vec <vec_file_name>`

Then, we need to provide the command with the positive image, which will be used to create multiple other positive images by combining them with the negatives:
`-img <image_file_name>`

After that, we need to provide the negative images list:
`-bg <background_file_name>`

All the negative images will be used as a background for randomly distorted versions of the targeted object (the positive image).

We also need to say how many new positives we want to generate:
`-num <number_of_samples>`
 
Also, we need to set the background color (currently grayscale images are assumed)
The background color denotes the transparent color. Since there might be compression artifacts, the amount of color tolerance can be specified by `-bgthresh`. All pixels withing `bgcolor-bgthresh` and `bgcolor+bgthresh` range are interpreted as transparent.
`-bgcolor <background_color>`
`-bgthresh <background_color_threshold>`

We should also define the maximal intensity deviation of pixels in foreground (positive) samples:
`-maxidev <max_intensity_deviation>`

And also the maximum rotation angles (in radians) for the foreground (positive) samples:
`-maxxangle <max_x_rotation_angle>`
`-maxyangle <max_y_rotation_angle>`
`-maxzangle <max_z_rotation_angle>`

At last, we need to define the output samples width and height (in pixels):
`-w <sample_width>`
`-h <sample_height>`

There are also other arguments (which we won't use), but you may check them out in the opencv <a href="https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html#positive-samples">docs</a>.

Our command for creating new positives should look like this:

```bash
opencv_createsamples -vec samples_<number>.vec -img <POSITIVE_IMAGE> -bg neg.txt -num 150 -bgcolor 0 -bgthresh 0 -maxxangle 1.1 -maxyangle 1.1 -maxzangle 0.5 -maxidev 40 -w 500 -h 500
```

Just make sure to put all the newly generated `samples_<number>.vec` files into a directory called `samples`.

The overall testing directory structure, after this step should be:

```txt
.
├── neg
|	|── neg_img_001.jpg
|	|── ...
|	└── neg_img_NNN.jpg
├── pos
|	|── pos_img_001.jpg
|	|── ...
|	└── pos_img_NNN.jpg
├── samples
|	|── samples_001.vec
|	|── ...
|	└── samples_NNN.vec
├── neg.txt
└── pos.txt
```

### Creating list of samples
After creating samples with all the positive images we had, it's time to create a `samples.txt` containing a list of all the `.vec` files, so we can (later) merge all the `samples.vec` files into one.

We shall do that the same way we created the `pos.txt` and `neg.txt`:

```bash
find samples/ -iname "*.vec" > samples.txt
```

The directory structure now is:

```txt
.
├── neg
|	|── neg_img_001.jpg
|	|── ...
|	└── neg_img_NNN.jpg
├── pos
|	|── pos_img_001.jpg
|	|── ...
|	└── pos_img_NNN.jpg
├── samples
|	|── samples_001.vec
|	|── ...
|	└── samples_NNN.vec
├── neg.txt
├── pos.txt
└── samples.txt
```

### Creating samples.vec containing all the generated samples











# Python script thath automates the process


preparing_samples.py -p <positives_dir> -n <negatives_dir> -num <number_of_new_positive_samples_to_be_created> -bgcolor <background_color> -bgthresh <background_color_threshold> -maxxangle <max_x_rotation_angle> -maxyangle <max_y_rotation_angle> -maxzangle <max_z_rotation_angle> -maxidev <max_intensity_deviation> -w <images_width> -h <images_height>"



  

COMMAND_FORMAT = "Error! The command should be: preparing_samples.py -p <positives_dir> -n <negatives_dir> -num <number_of_new_positive_samples_to_be_created> -bgcolor <background_color> -bgthresh <background_color_threshold> -  <max_x_rotation_angle> -maxyangle <max_y_rotation_angle> -maxzangle <max_z_rotation_angle> -maxidev <max_intensity_deviation> -w <images_width> -h <images_height>"





python3 preparing_samples_v0.0.1.py -p pos/ -n neg/ -num 10 -bgcolor 0 -bgthresh 0 -maxxangle 1.1 -maxyangle 1.1 -maxzangle 0.5 -maxidev 40 -w 500 -h 500






python3 preparing_samples_v0.0.1.py -p p/ -n n/ --num=10 --bgcolor=0 --bgthresh=0 --maxxangle=1.1 --maxyangle=1.1 --maxzangle=0.5 --maxidev=40 --width=500 --height=500