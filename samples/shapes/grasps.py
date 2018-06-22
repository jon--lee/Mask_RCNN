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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--weight_decay', required=False, type=float, default=0.0001)
ap.add_argument('--backbone', required=False, type=str, default='resnet101')

args = vars(ap.parse_args())


# Directory to save logs and trained model
NFS = '/nfs/diskstation/jonathan/mask'
MODEL_DIR = os.path.join(NFS, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class GraspsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "grasps"

    BACKBONE = args['backbone']

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    WEIGHT_DECAY = args['weight_decay']
    
config = GraspsConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


key_map = {
    'scrap': 1,
    'tape': 2,
    'tube': 3,
    'screwdriver': 4
}

key_map_reverse = {
    1: 'scrap',
    2: 'tape',
    3: 'tube',
    4: 'screwdriver',
}


class GraspDataset(utils.Dataset):

    def load_grasps(self, dataset_dir):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("grasp", 0, "scrap")
        self.add_class("grasp", 1, "tape")
        self.add_class("grasp", 2, "tube")
        self.add_class("grasp", 3, "screwdriver")

        annotations_dir = os.path.join(dataset_dir, 'annotations')
        images_dir = os.path.join(dataset_dir, 'images')
        
        anno_filenames = sorted([fn for fn in os.listdir(annotations_dir) if fn.endswith('.json')])
        images_filenames = sorted([fn for fn in os.listdir(images_dir) if fn.endswith('.png')])
        
        for i, (anno_name, img_name) in enumerate(zip(anno_filenames, images_filenames)):
            anno_path = os.path.join(annotations_dir, anno_name)
            img_path = os.path.join(images_dir, img_name)
            f = open(anno_path)
            anno = json.load(f)
            f.close()
            
            polygons = [r['points'] for r in anno['shapes']]
            label_ids = [key_map[r['label']] for r in anno['shapes']]
            image = skimage.io.imread(img_path)
            height, width = image.shape[:2]
                    
            self.add_image(
                'grasp',
                image_id=i,  # use file name as a unique image id
                path=img_path,
                width=width, height=height,
                polygons=polygons, label_ids=label_ids)
        
        
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        label_ids = np.array(info['label_ids'])
        
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            p = np.array(p)
            all_y = p[:, 1]
            all_x = p[:, 0]
            rr, cc = skimage.draw.polygon(all_y, all_x)
            mask[rr, cc, i] = 1
            

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), label_ids

        

dataset_train = GraspDataset()
dataset_train.load_grasps('/nfs/diskstation/jonathan/mask/dataset/dataset')
dataset_train.prepare()

dataset_val = GraspDataset()
dataset_val.load_grasps('/nfs/diskstation/jonathan/mask/dataset/dataset_val')
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
                       
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=30, 
            layers='heads')

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=15, 
            layers="all")


model_path = os.path.join(MODEL_DIR, "mask_rcnn_grasps.h5")
model.keras_model.save_weights(model_path)
