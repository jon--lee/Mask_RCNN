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
import pickle

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

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



# # Directory to save logs and trained model
model_dir = '/nfs/diskstation/jonathan/mask/logs/'
model_path = os.path.join(model_dir, "grasps_rgb/mask_rcnn_grasps_0029.h5")
dataset_dir = "/nfs/diskstation/jonathan/mask/dataset/dataset"
dataset_val_dir = "/nfs/diskstation/jonathan/mask/dataset/dataset_val"

image_dir = os.path.join(dataset_dir, "images/")
image_val_dir = os.path.join(dataset_val_dir, "images/")
anno_dir = os.path.join(dataset_dir, "annotations/")
anno_val_dir = os.path.join(dataset_val_dir, "annotations/")

output_dir = './output/output_rgb/'
output_results_dir = './output/output_rgb_results/'


def load_gt_masks(image_name, image_id, anno_path, image_shape):
    f = open(anno_path, 'r')
    anno = json.load(f)
    f.close()

    height, width = image_shape[:2]
    polygons = [r['points'] for r in anno['shapes']]
    label_ids = [key_map[r['label']] for r in anno['shapes']]

    gt_annos = []

    for i, (p, label_id) in enumerate(zip(polygons, label_ids)):
        p = np.array(p)
        all_y = p[:, 1]
        all_x = p[:, 0]
        rr, cc = skimage.draw.polygon(all_y, all_x)

        gt_mask = np.zeros((height, width))
        gt_mask[rr, cc] = 1
        gt_mask = gt_mask.astype(bool)

        gt_anno = {
            'image_name': image_name,
            'image_id': image_id,
            'category_id': label_id,
            'segmentation': gt_mask.copy(),
        }
        gt_annos.append(gt_anno)

    return gt_annos


def load_pred_masks(image_name, image_id, detection_results):
    r = detection_results[0]
    masks = r['masks'].copy()
    shape = masks.shape
    class_ids = r['class_ids']
    scores = r['scores']
    
    pred_annos = []

    for i in range(shape[2]):
        pred_mask = masks[:, :, i].astype(np.bool)
        class_id = class_ids[i]

        pred_anno = {
            'image_name': image_name,
            'image_id': image_id,
            'category_id': class_id,
            'segmentation': pred_mask.copy(),
            'score': scores[i]
        }
        pred_annos.append(pred_anno)

    return pred_annos


def save_results(filename, output_results_dir, pred_results, gt_results):
    assert filename.endswith(".pkl")
    pred_dir = os.path.join(output_results_dir, 'pred')
    gt_dir = os.path.join(output_results_dir, 'gt')

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)

    pred_path = os.path.join(pred_dir, filename)
    gt_path = os.path.join(gt_dir, filename)

    print("Saving pred output results to: " + pred_path)
    print("Saving gt output results to: " + gt_path)

    pred_file = open(pred_path, 'wb')
    pickle.dump(pred_results, pred_file, protocol=2)
    pred_file.close()
    gt_file = open(gt_path, 'wb')
    pickle.dump(gt_results, gt_file, protocol=2)
    gt_file.close()



def run_eval(model_dir, model_path, image_dir, image_val_dir, 
            output_dir, output_results_dir, anno_dir, anno_val_dir):

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

    if not os.path.exists(output_val):
    	os.makedirs(output_val)
    if not os.path.exists(output_train):
    	os.makedirs(output_train)

    def eval(names, src_dir, dst_dir, anno_dir=None):
        for filename in names:
            srcpath = os.path.join(src_dir, filename)
            dstpath = os.path.join(dst_dir, filename)

            print("Loading image from: " + srcpath)
            print("Saving image to: " + dstpath)

            image = skimage.io.imread(srcpath)
            image = skimage.color.grey2rgb(image)
            results = model.detect([image], verbose=1)

            r = results[0]
            figsize=(16, 16)
            _, ax = plt.subplots(1, figsize=figsize)
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, r['scores'], ax=ax)

            plt.savefig(dstpath)
            plt.close()
            plt.clf()
            plt.cla()


            if not anno_dir is None:
                image_id = int(filename[:-4][-4:])
                anno_filename = filename[:-4] + '.json'
                anno_path = os.path.join(anno_dir, anno_filename)

                pred_results = load_pred_masks(filename, image_id, results)
                gt_results = load_gt_masks(filename, image_id, anno_path, image.shape)
                results_filename = filename[:-4] + '.pkl'

                save_results(results_filename, output_results_dir, pred_results, gt_results)



    # eval(filenames[:10], image_dir, output_train)
    # eval(filenames[-10:], image_dir, output_train)
    eval(filenames_val[:], image_val_dir, output_val, anno_dir=anno_val_dir)

    exit()


if __name__ == '__main__':
    run_eval(model_dir, model_path, image_dir, image_val_dir, 
                output_dir, output_results_dir, anno_dir, anno_val_dir)
