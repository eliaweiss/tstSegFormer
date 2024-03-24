# train.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3 #100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 1280 original
IMAGE_WIDTH = 240 # 1918 original
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"    
    
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop= tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
    
    # forward
    with torch.cuda.amp.autocast():
        predictions =  model(data)
        loss = loss_fn(predictions, targets)
        
        
    # backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # update tqdm
    loop.set_postfix(loss=loss.item())
        
        
        
    
    
def main():
    pass

if __name__ == "__main__":
    main()


