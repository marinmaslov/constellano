#!/usr/bin/env python
""" Constellano Model Retrainer
Python script for object detection model retraining using Tensorflow.

Command format:
    py Retrainer.py --model <path_to_model> --pipeline <path_to_pipeline>
"""

# Imports
import os
import sys
import getopt

import tensorflow as tf

from datetime import datetime

from object_detection import model_lib_v2
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2

# Define TensorFlow versions
tf = tf.compat.v2

__author__ = "Marin Maslov"
__license__ = "MIT Licence"
__version__ = "2.1.2"
__maintainer__ = "Marin Maslov"
__email__ = "mmaslo00@fesb.hr"
__status__ = "Stable"

# Command format
COMMAND_FORMAT = "Error! The command should be: py Retrainer.py --model <path_to_model> --pipeline <path_to_pipeline>"

# Number of train steps.
TRAIN_STEPS = 10000

# Whether the job is executing on a TPU.
USE_TPU = False

# Integer defining how often we checkpoint.
CHECKPOINT_EVERY_N = 1000

# Whether or not to record summaries defined by the model or the training pipeline. This does not impact the summaries of the loss values which are always recorded.
RECORD_SUMMARIES = True


def convertClassesToLabelMap(classes, start=1):
    msg = ''
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + "\tid: " + str(id) + "\n"
        msg = msg + "\tname: '" + name + "'\n}\n\n"
    return msg[:-1]

def main(argv):
    model = ''
    pipeline = ''

    try:
        opts, args = getopt.getopt(argv, "h", ["model=", "pipeline="])
    except getopt.GetoptError:
        print(COMMAND_FORMAT)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(COMMAND_FORMAT)
            sys.exit()
        elif opt in ("--model"):
            model = arg
        elif opt in ("--pipeline"):
            pipeline = arg

    start_time = datetime.now()

    # Retrain the model
    tf.config.set_soft_device_placement(True)
    strategy = tf.compat.v2.distribute.MirroredStrategy()

    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path = pipeline,
            model_dir = model,
            train_steps = TRAIN_STEPS,
            use_tpu = USE_TPU,
            checkpoint_every_n = CHECKPOINT_EVERY_N,
            record_summaries = RECORD_SUMMARIES)

    # Export the model
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    text_format.Merge(FLAGS.config_override, pipeline_config)
    exporter_lib_v2.export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_dir,
        FLAGS.output_directory, FLAGS.use_side_inputs, FLAGS.side_input_shapes,
        FLAGS.side_input_types, FLAGS.side_input_names)
    """

    print("------------------------------------")
    end_time = datetime.now()
    print("[INFO]\tTotal execution time: " + str(end_time - start_time) + ".")


if __name__ == "__main__":
    main(sys.argv[1:])