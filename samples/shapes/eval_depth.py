import matplotlib
matplotlib.use('Agg')
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import skimage
import json
import matplotlib
import matplotlib.pyplot as plt
import IPython
from eval_rgb import run_eval

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# # Directory to save logs and trained model
model_dir = '/nfs/diskstation/jonathan/mask/logs/'
model_path = os.path.join(model_dir, "grasps_depth/mask_rcnn_grasps_0029.h5")
dataset_dir = "/nfs/diskstation/jonathan/mask/dataset/dataset_depth"
dataset_val_dir = "/nfs/diskstation/jonathan/mask/dataset/dataset_depth_val"

image_dir = os.path.join(dataset_dir, "images/")
image_val_dir = os.path.join(dataset_val_dir, "images/")
anno_dir = os.path.join(dataset_dir, "annotations/")
anno_val_dir = os.path.join(dataset_val_dir, "annotations/")

output_dir = './output/output_depth/'
output_results_dir = './output/output_depth_results/'


if __name__ == '__main__':
    run_eval(model_dir, model_path, image_dir, image_val_dir, 
                output_dir, output_results_dir, anno_dir, anno_val_dir)


