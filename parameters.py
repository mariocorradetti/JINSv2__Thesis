
#!/usr/bin/env python
import os
mode = 'train'  # choose between 'train' and 'test'  (maybe add evaluation too)

## --PARAMETERS-- ##
BASE_DIR = 'C:/Users/corra/OneDrive/Desktop/Blender' # the directory of this file
DATASET_DIR = BASE_DIR + '/coco'
masks_dir = '/{}_masks/'.format(mode)  # ???? no need to prepend the full path since it is already specified in the blender node
image_dir = DATASET_DIR + '/{}/'.format(mode)
ANN_DIR = DATASET_DIR + '/annotations'
image_id = id  # MI SERVE PER DIFFERENZIARE LE IMMAGINI AD OGNI ITERAZIONE

basedir = DATASET_DIR

LABEL_NAMES = ['antenna', 'body', 'solarPanel', 'earth']
EMPTY_LABEL = 0

IMAGE_SIZE = (700, 700)
CAM_FOCAL = 25
MAX_CAMERA_ANGLE = 35
MIN_CAMERA_DISTANCE = 8
MAX_CAMERA_DISTANCE = 14
NUM_LIGHTS, NUM_CAMERAS = (1, 1)

area_threshold = 1  # masks with a lower area value are discarded
