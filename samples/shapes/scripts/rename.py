import os
import argparse

ap = argparse.ArgumentParser()

# ap.add_argument('--input_dir', required=True, type=str)
# ap.add_argument('--images_output_dir', required=True, type=str)
# ap.add_argument('--annos_output_dir', required=True, type=str)
dataset_dir = 'dataset2/'
images_dir = os.path.join(dataset_dir, 'images')
annos_dir = os.path.join(dataset_dir, 'annotations')

import shutil

source_dir = annos_dir
sourcefiles = os.listdir(source_dir)
for i, file in enumerate(sorted(sourcefiles)):
    if file.endswith('.png'):
        c = 4
    elif file.endswith('.json'):
        c = 5

    j = 100 + i
    num = str(j).rjust(4, '0')
    print(num)

    new_file = list(file[:])
    new_file[-(c + 4):-c] = num
    new_file = ''.join(new_file)

    source_path = os.path.join(source_dir, file)
    dest_path = os.path.join(source_dir, new_file)

    os.rename(source_path, dest_path)

    print(source_path)
    print(dest_path)
    print()

