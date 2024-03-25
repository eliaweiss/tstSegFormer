import os
import random
from shutil import move
import glob
from os.path import join, isfile, splitext

BASE_PATH = "/home/ubuntu/work/carvana/segFormer"

train_path = f"{BASE_PATH}/train/"
val_path = f"{BASE_PATH}/valid/"
test_path = f"{BASE_PATH}/test/"

train_images = os.listdir(train_path)
train_images = list(filter(lambda x: "jpg" in x, train_images))

split_size = int(0.1 * len(train_images))  # % of the data for validation

val_test_images = random.sample(train_images, split_size)
val_images = val_test_images[:split_size//2]
test_images = val_test_images[split_size//2:]

################################
# Match masks with corresponding images
val_masks = []
for image in val_images:
    # Extract mask base name
    mask_base_name = image.split(".")[0]
    # Loop through validation images to find match
    val_masks.append(f"{mask_base_name}_mask.gif")
    
val_masks.sort()
val_images.sort()

for image, mask in zip(val_images, val_masks):
    mask_base_name = image.split(".")[0]
    assert mask_base_name in mask

    move(os.path.join(train_path, image), os.path.join(val_path, image))
    move(os.path.join(train_path, mask), os.path.join(val_path, mask))
    # break

################################

# Match masks with corresponding images
test_masks = []
for image in test_images:
    # Extract mask base name
    mask_base_name = image.split(".")[0]
    # Loop through testidation images to find match
    test_masks.append(f"{mask_base_name}_mask.gif")
    
test_masks.sort()
test_images.sort()

for image, mask in zip(test_images, test_masks):
    mask_base_name = image.split(".")[0]
    assert mask_base_name in mask

    move(os.path.join(train_path, image), os.path.join(test_path, image))
    move(os.path.join(train_path, mask), os.path.join(test_path, mask))
    # break