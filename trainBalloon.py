# train.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from unetModel.unet_model import UNET
# from DiceLoss import DiceLoss
from unetModel.utils import (
    get_loaders_balloon,
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)
import os
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

BASE_PATH = "/Users/eliaweiss/work/tstSegFormer/Balloons-1"

CHECKPOINT_PATH = "model_cp/balloon_checkpoint.pth.tar"

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("DEVICE", DEVICE)
BATCH_SIZE = 8
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 original
IMAGE_WIDTH = 240  # 1918 original
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = f"{BASE_PATH}/train/"
VAL_IMG_DIR = f"{BASE_PATH}/valid/"



def main():
    train_transform = A.Compose(
        [
            A. Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A. Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loaders, val_loaders = get_loaders_balloon(
        TRAIN_IMG_DIR,
        VAL_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)
        # change LOAD_MODEL to True
        check_accuracy(val_loaders, model, device=DEVICE)

    
    trainer = pl.Trainer(
        accelerator="auto", 
        devices="auto",
        max_epochs=NUM_EPOCHS,
        # precision=16
        )
    trainer.fit(model, train_loaders, val_loaders)
    
    trainer.validate(model, val_loaders)
    # model.predict_step(
    
    # torchvision.utils.save_image(y.unsqueeze(1), 
    #                                 os.path.join(folder,f"correct_{idx}.png")

if __name__ == "__main__":
    main()
