#!/usr/bin/env python
import json
import os
import random
import skimage
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # note: even if imported as PIL, the module name to be installed is 'pillow'
from pycocotools.coco import COCO

import argparse

parser = argparse.ArgumentParser()
#-db DATABASE  -n NUMBER OF PICS TO DISPLAY
parser.add_argument("-db", "--database", dest="dbase", default="train", help="train or test database")
parser.add_argument("-n", "--number", dest="num", default=5, help="number of random picks")
args = parser.parse_args()

try:
    args.num = int(args.num)
except ValueError as e:
    # print('Error: The argument to the -n flag must be a number')
    quit('Error: The argument to the -n flag must be a number')

if not (args.dbase == 'test' or args.dbase == 'train' or args.dbase == 'val'):
    quit('Error: The argument to the -db flag must either be "test" or "train" or "val"s.')

# ref:  https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
# mode = 'train'
mode = args.dbase
path = os.path.join(os.getcwd(), 'coco/')
images_path = os.path.join(path, '{}/'.format(mode))
im_list = []

for file in os.listdir(images_path):
    if file.endswith(".png"):  #the script doesn't work with .tif
        im_list.append(file)

#plt.ion() # enables interactive mode
# for im_name in im_list:
# for i in range(int(args.num)):
#for im_name in random.sample(im_list, min(args.num, len(im_list))):
for im_name in random.sample(im_list, min(1, 4)):

    # im_name = random.choice(im_list)
    im = os.path.join(images_path, im_name)
    im = np.array(Image.open(im), dtype=np.uint16)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch (takes top left corner, width and height as inputs)
    with open(os.path.join(path, 'annotations/instances_{}.json'.format(mode))) as json_file:
        data = json.load(json_file)
        # print (data)
        for item in data['images']:
            if item['file_name'] == im_name:
                image_id = item['id']
                ann_list = []
                for ann_item in data['annotations']:
                    if ann_item['image_id'] == image_id:
                        bbox = ann_item['bbox']
                        x, y, w, h = bbox
                        # x = bbox[0]
                        # y = bbox[1]
                        # w = bbox[2]
                        # h = bbox[3]
                        rect = patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor='r', facecolor='none', label='label')
                        # Add the patch to the Axes
                        ax.add_patch(rect)
                        # find the label corresponding to the annotation and write it on the image
                        cat_id = ann_item['category_id']
                        for cat in data['categories']:
                            if cat['id'] == cat_id:
                                label = cat['name']
                        labelx, labely = rect.get_xy()
                        ax.annotate(label, (labelx, labely), color='w', backgroundcolor="r", weight='bold',
                                    fontsize=6, ha='left', va='baseline')

                        annFile = os.path.join(path, 'annotations/instances_{}.json'.format(mode))
                        coco = COCO(annFile)
                        catIds = coco.getCatIds()
                        imgIds = coco.getImgIds(catIds=catIds)
                        #imgIds = coco.getImgIds(imgIds = imgIds[0])
                        imgIds = coco.getImgIds(imgIds=imgIds)
                        img = coco.imgs[image_id]
                        I = skimage.io.imread(os.path.join(images_path, im_name))
                        plt.imshow(I)
                        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                        anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.draw()

plt.show()  # not needed if using interactive mode
input('Press any key to close all figures...')
plt.close(fig='all')
