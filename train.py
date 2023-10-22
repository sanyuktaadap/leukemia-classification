import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

# Local Imports
from dataset import LeukemiaDataset
from model import LeukemiaClassifier
from utils import run_epoch

# Constants
TRAIN_IMG_PATH = "./data-split/train"
VAL_IMG_PATH = "./data-split/val"
CKPT_PATH = "./checkpoints/"
IMG_SIZE = (224, 224)
N_CLASSES = 4

# Hyperparams
lr = 0.01
momentum = 0.9
batch_size = 16
n_epochs = 50

# Load Train data
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=IMG_SIZE),
    v2.RandomRotation(degrees=20),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float, scale=True),
])

train_dataset = LeukemiaDataset(imgs_path=TRAIN_IMG_PATH, transforms=train_transforms)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

# Load Val data
val_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=IMG_SIZE),
    v2.ToDtype(torch.float, scale=True),
])

val_dataset = LeukemiaDataset(imgs_path=VAL_IMG_PATH, transforms=val_transforms)
val_dataloader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = LeukemiaClassifier()
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
logger = SummaryWriter()

model = model.to(device)

global_step = 0

# Training
for i in range(n_epochs):
    model.train()
    # run one epoch of training
    train_metrics = run_epoch(
        train_dataloader,
        model, 
        device, 
        loss_fn,
        logger, 
        opt=opt,
        step=i*len(train_dataloader)
    )
    
    model.eval()    
    # run one epoch of validation    
    val_metrics = run_epoch(
        val_dataloader,
        model, 
        device, 
        loss_fn,
        logger, 
        to_run="val", 
        step=i*len(val_dataloader)
    )
    
    print(f"Epoch {i}:")
    print(
        f"Train: Loss - {train_metrics[0]}, " + 
        f"Accuracy - {train_metrics[1]}, " +
        f"Specificity - {train_metrics[2]}, " +
        f"Precision - {train_metrics[3]}, " +
        f"Recall - {train_metrics[4]}, " +
        f"F1 - {train_metrics[5]}"
    )
    
    print(
        f"Val: Loss - {val_metrics[0]}, " + 
        f"Accuracy - {val_metrics[1]}, " +
        f"Specificity - {val_metrics[2]}, " +
        f"Precision - {val_metrics[3]}, " +
        f"Recall - {val_metrics[4]}, " +
        f"F1 - {val_metrics[5]}"
    )    

    torch.save(model.state_dict(), os.path.join(CKPT_PATH, f"checkpoint{i}.pt"))    