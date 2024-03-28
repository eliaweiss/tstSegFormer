#  createTotDS.py
import os
import random
from shutil import copy
import glob
from os.path import join, isfile, splitext

BASE_PATH = "/home/ubuntu/work/pdfBLines/datasetCrop"

train_path = f"{BASE_PATH}/train/"
toy_path = f"{BASE_PATH}/toy/"
try:
    os.makedirs(toy_path)
except FileExistsError:
    pass

train_images = os.listdir(train_path)
train_images = list(filter(lambda x: "jpg" in x, train_images))

split_size = 50 #int(0.1 * len(train_images))  # % of the data for validation

toy_images = random.sample(train_images, split_size)


################################
# Match masks with corresponding images
toy_masks = []
for image in toy_images:
    # Extract mask base name
    mask_base_name = image.split(".")[0]
    # Loop through validation images to find match
    toy_masks.append(f"{mask_base_name}_mask.png")
    
toy_masks.sort()
toy_images.sort()

for image, mask in zip(toy_images, toy_masks):
    mask_base_name = image.split(".")[0]
    assert mask_base_name in mask

    copy(os.path.join(train_path, image), os.path.join(toy_path, image))
    copy(os.path.join(train_path, mask), os.path.join(toy_path, mask))
    # break
