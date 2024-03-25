import os
import random
from shutil import move


train_path = "/home/ubuntu/work/carvana/data/train/"  
train_mask_path = "/home/ubuntu/work/carvana/data/train_masks/"
val_path = "/home/ubuntu/work/carvana/data/val/"  
val_mask_path = "/home/ubuntu/work/carvana/data/val_masks/"  

train_images = os.listdir(train_path)
train_masks = os.listdir(train_mask_path)

val_size = int(0.1 * len(train_images))  # 10% of the data for validation
val_images = random.sample(train_images, val_size)
# Match masks with corresponding images
# val_masks = [mask for mask in train_masks if mask.split(".")[0] in [image.split(".")[0] for image in val_images]]  
val_masks = []
for mask in train_masks:
  # Extract mask base name
  mask_base_name = mask.split(".")[0]
  # Loop through validation images to find match
  for image in val_images:
    image_base_name = image.split(".")[0]
    # Check if mask base name matches image base name with "_mask" appended
    if mask_base_name == image_base_name + "_mask":
      val_masks.append(mask)
      break  # Exit inner loop once corresponding mask is found
val_masks.sort()
val_images.sort()

for image, mask in zip(val_images,val_masks):
    mask_base_name = image.split(".")[0]
    assert mask_base_name in mask
    
    move(os.path.join(train_path, image), os.path.join(val_path, image))
    move(os.path.join(train_mask_path, mask), os.path.join(val_mask_path, mask))