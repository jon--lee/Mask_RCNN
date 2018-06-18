import numpy as np
import pickle
import IPython
import sys
import os
from stats_rgb import encode_gt, encode_pred, compute_coco_metrics
import json


root_dir = os.path.abspath("../../")
sys.path.append(root_dir)  # To find local version of the library

results_dir = os.path.join(root_dir, 'samples/shapes/output/output_depth_results')
results_gt_dir = os.path.join(results_dir, 'gt')
results_pred_dir = os.path.join(results_dir, 'pred')



if __name__ == '__main__':
    encode_gt(results_gt_dir)
    encode_pred(results_pred_dir)
    compute_coco_metrics(results_gt_dir, results_pred_dir)
