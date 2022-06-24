#!/usr/bin/env python
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
# from pycococreatortools import pycococreatortools
import pycococreatortools

#import parameters
import paramoon

# ROOT_DIR = 'train'
# IMAGE_DIR = os.path.join(ROOT_DIR, "satellites_train2020")
# MASKS_DIR = os.path.join(ROOT_DIR, "annotations")
ROOT_DIR = parameters.DATASET_DIR
# IMAGE_DIR = (parameters.TRAIN_DIR, parameters.TEST_DIR)
# MASKS_DIR = parameters.MASKS_DIR

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'antenna',
        'supercategory': 'part',
    },
    {
        'id': 2,
        'name': 'body',
        'supercategory': 'part',
    },
    {
        'id': 3,
        'name': 'solarPanel',
        'supercategory': 'part',
    },
    {
        'id': 4,
        'name': 'earth',
        'supercategory': 'part',
    },
 #   {
 #       'id': 5,
 #       'name': 'thruster',
 #       'supercategory': 'part',
 #   },
]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']    #  suffix of the file
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types]) # match the suffix of the files with the suffix in file_types?
    files = [os.path.join(root, f) for f in files] #?
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename): #?
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main(image_dir, masks_dir):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(image_dir):
        image_files = filter_for_jpeg(root, files)   # select all images with desired suffix ?

        # go through each image
        for image_filename in image_files: # ciclo su tutti i file trovati
            image = Image.open(image_filename) #Opens and identifies the given image file.
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size) # genero una lista contenenti proprietà dell'immagine
            coco_output["images"].append(image_info) #appendo "images" alla lista appena creata (?)

            # filter for associated png annotations
            for root, _, files in os.walk(masks_dir): # passo la dir delle maschere e restituisce i file all'interno(non quelli all interno di subdir) e il nome della cartella(root)
                annotation_files = filter_for_annotations(root, files, image_filename) #??

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    # if annotation_info is not None:  # original code
                    #     coco_output["annotations"].append(annotation_info)

                    if annotation_info is None:  # modified code to remove the useless annotation image
                        os.remove(annotation_filename)
                        continue

                    coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    # TODO: delete file if it already exists
    with open('{}/instances_{}.json'.format(parameters.ANN_DIR, os.path.basename(os.path.normpath(image_dir))), 'w') as output_json_file:
        json.dump(coco_output, output_json_file, sort_keys=False, indent=4)


if __name__ == "__main__":
    if not os.path.exists(parameters.ANN_DIR):
#        os.makedirs(parameters.ANN_DIR)
        os.makedirs(parameters.ANN_DIR)
    for mode in ['train', 'test', 'val']:
        masks_dir = ROOT_DIR + '/{}_masks/'.format(mode)
        image_dir = ROOT_DIR + '/{}/'.format(mode)
        main(image_dir, masks_dir)