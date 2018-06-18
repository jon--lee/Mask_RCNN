import numpy as np
import pickle
import IPython
import sys
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
import json


root_dir = os.path.abspath("../../")
sys.path.append(root_dir)  # To find local version of the library

results_dir = os.path.join(root_dir, 'samples/shapes/output/output_rgb_results')
results_gt_dir = os.path.join(results_dir, 'gt')
results_pred_dir = os.path.join(results_dir, 'pred')

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


def encode_gt(results_gt_dir):
    gt_annos = {
        'images': [],
        'annotations': [],
        'categories': [
            {'name': 'scrap',
             'id': key_map['scrap'],
             'supercategory': 'scrap'},
             {'name': 'tape',
             'id': key_map['tape'],
             'supercategory': 'tape'},
             {'name': 'tube',
             'id': key_map['tube'],
             'supercategory': 'tube'},
             {'name': 'screwdriver',
             'id': key_map['screwdriver'],
             'supercategory': 'screwdriver'}
        ]
    }

    filenames = sorted([d for d in os.listdir(results_gt_dir) if d.endswith('pkl')])
    print len(filenames)

    for i, filename in enumerate(filenames):
        filepath = os.path.join(results_gt_dir, filename)
        f = open(filepath, 'rb')
        results = pickle.load(f)
        f.close()


        result = results[0]
        image_id = result['image_id']
        seg = result['segmentation']
        height, width = seg.shape
        image_name = result['image_name']
        im_anno = {
            'id': image_id,
            'width': width,
            'height': height,
        }

        gt_annos['images'].append(im_anno)

        for j, result in enumerate(results):
            bin_mask = result['segmentation'].astype(np.uint8)
            category_id = result['category_id']
            instance_id = i * 100 + (j + 1)

            def bbox2(img):
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                return int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)

            encode_mask = mask.encode(np.asfortranarray(bin_mask))
            encode_mask['counts'] = encode_mask['counts'].decode('ascii')
            size = int(mask.area(encode_mask))
            x, y, w, h = bbox2(bin_mask)

            instance_anno = {
                "id" : instance_id,
                "image_id" : image_id,
                "category_id" : category_id,
                "segmentation" : encode_mask,
                "area" : size,
                "bbox" : [x, y, w, h],
                "iscrowd" : 0,
            }
            gt_annos['annotations'].append(instance_anno)


    anno_path = os.path.join(results_gt_dir, 'annos_gt.json')
    json.dump(gt_annos, open(anno_path, 'w+'))
    print "successfully wrote GT annotations to", anno_path
    return gt_annos




def encode_pred(results_pred_dir):
    annos = []

    filenames = sorted([d for d in os.listdir(results_pred_dir) if d.endswith('pkl')])
    print len(filenames)

    for i, filename in enumerate(filenames):
        filepath = os.path.join(results_pred_dir, filename)
        f = open(filepath, 'rb')
        results = pickle.load(f)
        f.close()

        for j, result in enumerate(results):

            bin_mask = result['segmentation'].astype(np.uint8)
            image_id = result['image_id']
            category_id = result['category_id']
            score = result['score']

            encode_mask = mask.encode(np.asfortranarray(bin_mask))
            encode_mask['counts'] = encode_mask['counts'].decode('ascii')

            anno = {
                'image_id': int(image_id),
                'category_id': int(category_id),
                'score': float(score),
                'segmentation': encode_mask 
            }  
            annos.append(anno)


    anno_path = os.path.join(results_pred_dir, 'annos_pred.json')
    json.dump(annos, open(anno_path, 'w+'))
    print "successfully wrote prediction annotations to", anno_path
    return annos


def compute_coco_metrics(results_gt_dir, results_pred_dir):
    """Given paths to two directories, one containing a COCO ground truth annotations
    file and the other a path to a COCO prediction annotations file, compute the COCO
    evaluation metrics on the predictions.
    Because the COCO API is weird and prints out summary values, we need this
    terrible hack to capture them from stdout.
    https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    """

    gt_path = os.path.join(results_gt_dir, 'annos_gt.json')
    pred_path = os.path.join(results_pred_dir, 'annos_pred.json')

    import io
    # from contextlib import redirect_stdout

    pred_annos = json.load(open(pred_path, 'r'))
    gt_annos = json.load(open(gt_path, 'r'))


    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(pred_path)

    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')

    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.params.useCats = False

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # f = io.StringIO()
    # with redirect_stdout(f):
    #     cocoEval.summarize()
    # out = f.getvalue()
    # print(out)




if __name__ == '__main__':
    encode_gt(results_gt_dir)
    encode_pred(results_pred_dir)
    compute_coco_metrics(results_gt_dir, results_pred_dir)
