import os
import pytorch_lightning as pl
from roboflow import Roboflow
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from unetModel.balloonDataset import BalloonDataset


class BalloonDataModule (pl.LightningDataModule):
    def __init__(self,
                data_dir,
                batch_size=32,
                num_workers=2,
                image_height=512,
                image_width=512,
                pin_memory=True,
                ):
        self.image_height = image_height
        self.image_width = image_width
        self.pin_memory = pin_memory
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dir = os.path.join(self.data_dir, "train")
        self.valid_dir = os.path.join(self.data_dir, "valid")

        self.train_transform = A.Compose(
            [
                A. Resize(height=image_height, width=image_width),
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
        self.val_transform = A.Compose(
            [
                A.Resize(height=image_height, width=image_width),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    def prepare_data(self):
        rf = Roboflow(api_key="HUBAAfRrsHWybDdGgXbg")
        project = rf.workspace("paul-guerrie-tang1").project("balloons-geknh")
        self.dataset = project.version(1).download("png-mask-semantic")

    def setup(self, stage):
        self.train_ds = BalloonDataset(
            root_dir=self.train_dir,
            transform=self.train_transform,
        )

        self.val_ds = BalloonDataset(
            root_dir=self.valid_dir,
            transform=self.val_transform,
        )

    def train_dataloader(self):
        self.train_loaders = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return self.train_loaders

    def val_dataloader(self):
        self.val_loaders = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return self.val_loaders

    def test_dataloader(self): pass
