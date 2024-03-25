# train.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modelUnet import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import os 
BASE_PATH = "/home/ubuntu/work/carvana"

CHECKPOINT_PATH = "model_cp/my_checkpoint.pth.tar"

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE",DEVICE)
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 original
IMAGE_WIDTH = 240  # 1918 original
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = f"{BASE_PATH}/train/"
TRAIN_MASK_DIR = f"{BASE_PATH}/train_masks/"
VAL_IMG_DIR = f"{BASE_PATH}/val/"
VAL_MASK_DIR = f"{BASE_PATH}/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

    # forward
    with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions, targets)

    # backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # update tqdm
    loop.set_postfix(loss=loss.item())


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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loaders, val_loaders = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)
    
    check_accuracy(val_loaders, model, device=DEVICE) # change LOAD_MODEL to True

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loaders, model, optimizer, loss_fn, scaler)

        # save model
        check_point = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(check_point, CHECKPOINT_PATH)

        # check accuracy
        check_accuracy(val_loaders, model, device=DEVICE)

        # print some example to folder
        save_predictions_as_imgs(
            val_loaders, model, folder="save_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
