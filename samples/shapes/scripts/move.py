import os
import argparse

# ap = argparse.ArgumentParser()

# ap.add_argument('--input_dir', required=True, type=str)
# ap.add_argument('--images_output_dir', required=True, type=str)
# ap.add_argument('--annos_output_dir', required=True, type=str)


# args = vars(ap.parse_args())

# if not os.path.exists(args['images_output_dir']):
#     os.makedirs(args['images_output_dir'])
# if not os.path.exists(args['annos_output_dir']):
#     os.makedirs(args['annos_output_dir'])

source_img_dir = 'dataset2/images/'
dest_img_dir = 'dataset_val/images/'

source_anno_dir = 'dataset2/annotations/'
dest_anno_dir = 'dataset_val/annotations/'

import shutil

source_images = sorted(os.listdir(source_img_dir))[:10]
source_annos = sorted(os.listdir(source_anno_dir))[:10]


for filename in source_images:
    if filename.endswith('.png'):
        source_path = os.path.join(source_img_dir, filename)
        dest_path = os.path.join(dest_img_dir, filename)

    elif filename.endswith('.json'):
        source_path = os.path.join(source_anno_dir, filename)
        dest_path = os.path.join(dest_anno_dir, filename)

    shutil.copyfile(source_path, dest_path)




