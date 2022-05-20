#!/usr/bin/env python
""" Constellano Tensorflow2 Model Retraining Script
Python script for model retraining using Tensorflow2 and Keras.
Command format:
    py Retrain.py --model <model_name> --images <images_dir> --output <output_dir_name>
"""

import os
import sys
import getopt

# import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from datetime import datetime

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.0.1"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

# Constants --- START
COMMAND_FORMAT = "Error! The command should be: py Retrain.py --model <model_name> --images <images_dir> --output <output_dir_name>"
DO_FINE_TUNING = False

MODEL_HANDLE_MAP = {
    "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
    "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
    "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
    "efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
    "efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
    "efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2",
    "efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
    "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
    "efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2",
    "efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2",
    "efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2",
    "efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
    "efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
    "efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
    "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
    "efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
    "efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
    "efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
    "efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
    "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
    "efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
    "efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
    "efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
    "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
    "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
    "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
    "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
    "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
    "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
    "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
    "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
    "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/1",
    "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
    "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5",
    "inception_openimages_v2": "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
    "inception_resnet_v2_fast_rcnn": "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1",
    "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/4",
    "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/feature-vector/4",
    "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/feature-vector/4",
    "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/4",
    "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature-vector/4",
    "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature-vector/4",
    "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
    "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
    "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
    "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
    "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
    "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
    "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
    "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
    "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
    "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}

MODEL_IMAGE_SIZE_MAP = {
    "efficientnetv2-s": 384,
    "efficientnetv2-m": 480,
    "efficientnetv2-l": 480,
    "efficientnetv2-b0": 224,
    "efficientnetv2-b1": 240,
    "efficientnetv2-b2": 260,
    "efficientnetv2-b3": 300,
    "efficientnetv2-s-21k": 384,
    "efficientnetv2-m-21k": 480,
    "efficientnetv2-l-21k": 480,
    "efficientnetv2-xl-21k": 512,
    "efficientnetv2-b0-21k": 224,
    "efficientnetv2-b1-21k": 240,
    "efficientnetv2-b2-21k": 260,
    "efficientnetv2-b3-21k": 300,
    "efficientnetv2-s-21k-ft1k": 384,
    "efficientnetv2-m-21k-ft1k": 480,
    "efficientnetv2-l-21k-ft1k": 480,
    "efficientnetv2-xl-21k-ft1k": 512,
    "efficientnetv2-b0-21k-ft1k": 224,
    "efficientnetv2-b1-21k-ft1k": 240,
    "efficientnetv2-b2-21k-ft1k": 260,
    "efficientnetv2-b3-21k-ft1k": 300, 
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 260,
    "efficientnet_b3": 300,
    "efficientnet_b4": 380,
    "efficientnet_b5": 456,
    "efficientnet_b6": 528,
    "efficientnet_b7": 600,
    "inception_v3": 299,
    "inception_resnet_v2": 299,
    "nasnet_large": 331,
    "pnasnet_large": 331,
}

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
# Constants --- END

# Function that builds the dataset
def buildDataset(images_dir, subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        images_dir, # Images directory
        validation_split = .20,
        subset = subset,
        label_mode = "categorical",
        # Seed needs to provided when using validation_split and shuffle = True.
        # A fixed seed is used so that the validation set is stable across runs.
        seed = 123,
        image_size = IMAGE_SIZE,
        batch_size = 1)

# Function that prepares the training dataset
def prepareTrainingDataset(images_dir):
    print("[INFO]\tPrepare the training dataset")

    # Prepare the training dataset
    training_dataset = buildDataset(images_dir, "training")

    # Get the label names
    label_names = tuple(training_dataset.class_names)

    training_size = training_dataset.cardinality().numpy()
    training_dataset = training_dataset.unbatch().batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()

    # Get the normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    # Get the preprocessing model
    preprocessing_model = tf.keras.Sequential([normalization_layer])

    # Map the training dataset and return it
    training_dataset = training_dataset.map(lambda images, labels: (preprocessing_model(images), labels))

    return training_dataset, label_names, training_size, normalization_layer

# Function that prepares the validation dataset
def prepareValidationDataset(images_dir, normalization_layer):
    print("[INFO]\tPrepare the validation dataset")

    # Prepare the training dataset
    validation_dataset = buildDataset(images_dir, "validation")

    validation_size = validation_dataset.cardinality().numpy()
    validation_dataset = validation_dataset.unbatch().batch(BATCH_SIZE)

    # Map the validation dataset and return it
    validation_dataset = validation_dataset.map(lambda images, labels: (normalization_layer(images), labels))

    return validation_dataset, validation_size

def convertClassesToLabelMap(classes, start=1):
    msg = ''
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + " id: " + str(id) + "\n"
        msg = msg + " name: '" + name + "'\n}\n\n"
    return msg[:-1]

# Main function
def main(argv):
    model_name = ''
    images_dir = ''
    output = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["model=", "images=", "output="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--model"):
            model_name = arg
        elif opt in ("--images"):
            images_dir = arg
        elif opt in ("--output"):
            if arg[-1] == "/":
                output = arg
            else:
                output = arg + "/"

    start_time = datetime.now()

    if not os.path.exists(output) != False:
        print("Creating directory: " + output)
        os.mkdir(output)

    #location = str(images_dir)
    #output = location + 'resized/'

    # PRINT TW VERSIONS
    print("[INFO]\tTF version:", tf.__version__)
    print("[INFO]\tHub version:", hub.__version__)
    print("[INFO]\tGPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    # SELECT THE MODEL AND IMAGE SIZE
    print("[INFO]\tSelecting " + str(model_name) + " model")
    model_handle = MODEL_HANDLE_MAP.get(model_name)
    pixels = MODEL_IMAGE_SIZE_MAP.get(model_name, 224)
    print("[INFO]\tSelected model: " + str(model_name) + ":" + str(model_handle))

    IMAGE_SIZE = (pixels, pixels)
    print("[INFO]\tInput size: " + str(IMAGE_SIZE))

    # BUILD THE TRAINING DATASET
    training_dataset, label_names, training_size, normalization_layer = prepareTrainingDataset(images_dir)

    # Save the label map
    label_file_path = output + "label_map.pbtxt"
    with open(label_file_path, 'w') as f:
        f.write(convertClassesToLabelMap(label_names))

    # BUILD THE VALIDATION DATASET
    validation_dataset, validation_size = prepareValidationDataset(images_dir, normalization_layer)

    # DEFINE AND CREATE THE MODEL
    print("[INFO]\tBuilding model with: " + str(model_handle))
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape = IMAGE_SIZE + (3,)),
        hub.KerasLayer(model_handle, trainable = DO_FINE_TUNING),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(len(label_names), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + IMAGE_SIZE+(3,))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy'])

    # SPECIFY THE CHECKPOINTS
    checkpoint_path = output + "variables/" + "variables-{epoch:04d}"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_path, 
        verbose = 1, 
        save_weights_only = True,
        save_freq = 1 * BATCH_SIZE)

    # MODEL RETRAINING
    steps_per_epoch = training_size // BATCH_SIZE
    validation_steps = validation_size // BATCH_SIZE
    """
    model.fit(
        training_dataset,
        #label_names, # Not needed when training on dataset!
        epochs = 5,
        steps_per_epoch = steps_per_epoch,
        validation_data = validation_dataset,
        validation_steps = validation_steps,
        #callbacks = [cp_callback],
        verbose = 0).history
    """

    # hist = 
    model.fit(
        training_dataset,
        epochs = 5,
        steps_per_epoch = steps_per_epoch,
        validation_data = validation_dataset,
        validation_steps = validation_steps,
        callbacks = [cp_callback]) # .history

    # IF WANTED: PRINT STATISTICS
    """
    
    plt.figure()
    plt.ylabel("Loss (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,2])
    plt.plot(hist["loss"])
    plt.plot(hist["val_loss"])
    plt.figure()
    plt.ylabel("Accuracy (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,1])
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])
    """
    
    # MODEL VALIDATION
    x, y = next(iter(validation_dataset))
    validation_image = x[0, :, :, :]
    true_index = np.argmax(y[0])

    # Expand the validation image to (1, 224, 224, 3) before predicting the label
    prediction_scores = model.predict(np.expand_dims(validation_image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("True label: " + label_names[true_index])
    print("Predicted label: " + label_names[predicted_index])

    # SAVE RETRAINED MODEL
    saved_model_path = output + model_name + "/" + f"model_{model_name}"
    tf.saved_model.save(model, saved_model_path)

    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])