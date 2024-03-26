# https://www.youtube.com/watch?v=IHq1t7NxS8k&ab_channel=AladdinPersson
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch.optim as optim
# import torchmetrics
from torchmetrics.classification import Accuracy
LEARNING_RATE = 1e-4
from torchmetrics import Metric

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("dice_score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        
    def update(self,preds, targets):
        preds = (preds > 0.5).float()
        
        
        self.correct +=  (preds == targets).sum()
        self.total += torch.numel(preds)
        self.dice_score += (2*(preds*targets).sum())/(
            (preds+targets).sum() + 1e-8
        )        
        

    def compute(self):
        # Calculate accuracy and dice coefficient
        accuracy = self.correct.float() / self.total.float()
        dice_score = self.dice_score.float()
        return {'accuracy': accuracy, 'dice_score': dice_score}
            
class DoubleConv(nn.Module):
    ################################################################
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # kernel, stride, padding - same convolution
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # kernel, stride, padding - same convolution
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    ################################################################
    def forward(self, x):
        return self.conv(x)


class UNET(pl.LightningModule):
    ################################################################
    def __init__(self,
                 in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512],
                 metrics_interval=100
                 ):
        super(UNET, self).__init__()
        self._init_unet(in_channels, out_channels, features)
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        self.my_accuracy = MyAccuracy()
        self.metrics_interval = metrics_interval
    ################################################################

    def _init_unet(self, in_channels, out_channels, features):
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        return in_channels
        
        # self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    ################################################################
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
    ################################################################
    def training_step(self, batch, batch_idx):
        loss , predictions, targets = self._common_step(batch, batch_idx)
        # self.log('train_loss', loss)
        if True: #batch_idx % self.metrics_interval == 0:
            accuracy = self.my_accuracy(predictions, targets)
            
            # metrics = self.train_mean_iou.compute(
            #     num_labels=self.num_classes,
            #     ignore_index=255,
            #     reduce_labels=False,
            # )

            metrics = {'loss': loss, "accuracy": accuracy["accuracy"], "dice_score": accuracy["dice_score"]}

            # for k,v in metrics.items():
            #     self.log(k,v)

        else:
            metrics = {'loss': loss}
        
        self.log_dict(metrics, prog_bar=True)
        

        return(metrics)
    
    ################################################################
    def validation_step(self, batch, batch_idx):
        loss , predictions, targets = self._common_step(batch, batch_idx)
        # self.log('val_loss', loss)
        # self.log_dict({'val_loss': loss, 'accuracy': accuracy})        
        return loss
    
    ################################################################
    def test_step(self, batch, batch_idx):
        loss , predictions, targets = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        
        return loss
    
    ################################################################
    def _common_step(self, batch, batch_idx):
        # Get data and targets
        data, targets = batch
        targets = targets.float().unsqueeze(1)

        # Forward pass with AMP (assuming AMP is enabled)
        # with torch.cuda.amp.autocast():
        predictions = self.forward(data)
        loss = self.loss_fn(predictions, targets)

        # Automatic backward pass and optimizer step (handled by Lightning)

        # Automatic logging (can be configured in trainer)
 
        return loss , predictions, targets   

    
    ################################################################
    def predict_step(self, batch):
        data, targets = batch
        # with torch.cuda.amp.autocast():
        preds = torch.sigmoid(self.forward(data))
        
        preds = (preds > 0.5).float()
        
        return preds
        
    
    ################################################################
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)
            
################################################################
def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    pred = model(x)
    print(pred.shape)
    print(x.shape)
    assert pred.shape == x.shape


if __name__ == "__main__":
    test()
