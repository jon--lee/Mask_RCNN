import os
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('--input_dir', required=True, type=str)
ap.add_argument('--images_output_dir', required=True, type=str)
ap.add_argument('--annos_output_dir', required=True, type=str)


args = vars(ap.parse_args())

if not os.path.exists(args['images_output_dir']):
    os.makedirs(args['images_output_dir'])
if not os.path.exists(args['annos_output_dir']):
    os.makedirs(args['annos_output_dir'])

import shutil

sourcepath=args['input_dir']
sourcefiles = os.listdir(sourcepath)
dest_images = args['images_output_dir']
dest_annos = args['annos_output_dir']
for file in sourcefiles:
    if file.endswith('.png'):
        shutil.copyfile(os.path.join(sourcepath,file), os.path.join(dest_images,file))
    elif file.endswith('.json'):
        shutil.copyfile(os.path.join(sourcepath,file), os.path.join(dest_annos,file))


