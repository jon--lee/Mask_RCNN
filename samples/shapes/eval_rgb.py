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
model_dir = os.path.join(ROOT_DIR, "logs")
model_path = os.path.join(model_dir, "grasps_rgb/mask_rcnn_grasps_0029.h5")
dataset_dir = os.path.join(ROOT_DIR, "samples/shapes/dataset")
dataset_val_dir = os.path.join(ROOT_DIR, "samples/shapes/dataset_val")

image_dir = os.path.join(dataset_dir, "images/")
image_val_dir = os.path.join(dataset_val_dir, "images/")
anno_dir = os.path.join(dataset_dir, "annotations/")
anno_val_dir = os.path.join(dataset_val_dir, "annotations/")

output_dir = './output/output_rgb/'
masks_dir = './output/output_rgb_masks/'


def load_masks(anno_path, image_shape):
    f = open(anno_path, 'r')
    anno = json.load(f)
    f.close()

    polygons = [r['points'] for r in anno['shapes']]
    height, width = image_shape[:2]
    masks = np.zeros((len(polygons), height, width))
    
    for i, p in enumerate(polygons):
        p = np.array(p)
        all_y = p[:, 1]
        all_x = p[:, 0]
        rr, cc = skimage.draw.polygon(all_y, all_x)
        masks[i, rr, cc] = 1

    masks = masks.astype(np.bool)

    return masks

def run_eval(model_dir, model_path, image_dir, image_val_dir, 
            output_dir, masks_dir, anno_dir, anno_val_dir):

    class GraspsConfig(Config):
        """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = "grasps"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 4  # background + 4 shapes

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

        BATCH_SIZE = 1

    config = GraspsConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    model.load_weights(model_path, by_name=True)


    class_names =  ['BG', 'scrap', 'tape', 'tube', 'screwdriver']
    filenames = sorted([filename for filename in list(os.walk(image_dir))[0][2] if filename.endswith('.png')])
    filenames_val = sorted([filename for filename in list(os.walk(image_val_dir))[0][2] if filename.endswith('.png')])

    random.shuffle(filenames)
    random.shuffle(filenames_val)

    output_val = os.path.join(output_dir, 'val/')
    output_train = os.path.join(output_dir, 'train/')
    masks_pred = os.path.join(masks_dir, 'pred/')
    masks_gt = os.path.join(masks_dir, 'gt/')

    if not os.path.exists(output_val):
    	os.makedirs(output_val)
    if not os.path.exists(output_train):
    	os.makedirs(output_train)
    if not os.path.exists(masks_pred):
        os.makedirs(masks_pred)
    if not os.path.exists(masks_gt):
        os.makedirs(masks_gt)

    def eval(names, src_dir, dst_dir, anno_dir=None):
        for filename in names:
            srcpath = os.path.join(src_dir, filename)
            dstpath = os.path.join(dst_dir, filename)

            print(srcpath)
            print(dstpath)


            image = skimage.io.imread(srcpath)
            image = skimage.color.grey2rgb(image)
            results = model.detect([image], verbose=1)
            r = results[0]
            figsize=(16, 16)
            _, ax = plt.subplots(1, figsize=figsize)
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, r['scores'], ax=ax)

            if not anno_dir is None:
                masks = r['masks'].copy()
                shape = masks.shape
                pred_masks = np.zeros((shape[2], shape[0], shape[1]))
                for i in range(shape[2]):
                    pred_masks[i, :, :] = masks[:, :, i]

                anno_filename = filename[:-4] + '.json'
                anno_path = os.path.join(anno_dir, anno_filename)
                gt_masks = load_masks(anno_path, image.shape)

                mask_filename = filename[:-4] + '.npy'
                pred_filepath = os.path.join(masks_pred, mask_filename)
                gt_filepath = os.path.join(masks_gt, mask_filename)

                np.save(pred_filepath, pred_masks)
                np.save(gt_filepath, gt_masks)




            plt.savefig(dstpath)
            plt.close()
            plt.clf()
            plt.cla()


    # eval(filenames[:10], image_dir, output_train)
    # eval(filenames[-10:], image_dir, output_train)
    eval(filenames_val[:], image_val_dir, output_val, anno_dir=anno_val_dir)

    exit()


if __name__ == '__main__':
    run_eval(model_dir, model_path, image_dir, image_val_dir, 
                output_dir, masks_dir, anno_dir, anno_val_dir)
