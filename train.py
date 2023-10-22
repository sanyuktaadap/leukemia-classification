from torch.utils.data import DataLoader

# Local Imports
from dataset import LeukemiaDataset
from utils import transforms
from utils import run_model

# Constants
TRAIN_IMG_PATH = "data-split/train"

# Hyperparams
lr = 0.01
momentum = 0.9
batch_size = 16
n_epochs = 50

# Load data
train_dataset = LeukemiaDataset(imgs_path=TRAIN_IMG_PATH, transforms=transforms)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

# Training
run_train = run_model(n_epochs=n_epochs,
                      dataloader=train_dataloader,
                      to_run="train")