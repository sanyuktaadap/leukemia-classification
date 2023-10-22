from torch.utils.data import DataLoader

# Local Imports
from dataset import LeukemiaDataset
from utils import run_model

# Constants
VAL_IMG_PATH = "data-split/val"

# Hyperparameters
n_epochs = 1
batch_size=None

# Load data
val_dataset = LeukemiaDataset(imgs_path=VAL_IMG_PATH, transforms=None)
val_dataloader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=2)

# Testing
run_val = run_model(n_epochs=n_epochs,
                    dataloader=val_dataloader,
                    to_run="val")