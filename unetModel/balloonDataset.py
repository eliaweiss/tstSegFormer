# https://www.kaggle.com/competitions/carvana-image-masking-challenge/data
# wget 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/6927/45059/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1711540363&Signature=H1%2FdxoQ6jSr0B3Is8Bk6QBiYWriJsy1dTdvWzZoUNA0kGTZErRVd2WfNXkYh0Ig8DGLnwAMR8P01cs9B2Qb76d17hin27dd8DeN4JUHpHwlccz22Vo7%2FL7Oe6kRC6xVI9C1P1Gc%2F0BpyJLWkL7811STCp3AY3G90I0g7eJ6ljMatipz3M8%2FuNUBV2F979%2BK%2B6%2FmnFpf5%2Fps4F4l9RxpB0FwWHuJolwhG3IPnarLohkj6yk9pWlriq%2B4SucgFsXWxc9scUvFGFMB086NlWH5YBRwoaM5653vp8CQhHUCtCYPkIF4D6Z9GQqSrcPsHzja1IoVv8S1%2FrKcslRTfNBHeuw%3D%3D&response-content-disposition=attachment%3B+filename%3Dcarvana-image-masking-challenge.zip'
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class BalloonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        image_file_names = [f for f in os.listdir(self.root_dir) if '.jpg' in f]
        mask_file_names = [f for f in os.listdir(self.root_dir) if '.png' in f]

        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)  
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(os.path.join(self.root_dir, self.images[idx])).convert("RGB"))
        mask =np.array(Image.open(os.path.join(self.root_dir, self.masks[idx])).convert("L"), dtype=np.float32)
        
        mask[mask == 255] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            
        return image,mask
        
        
        
        
        
        