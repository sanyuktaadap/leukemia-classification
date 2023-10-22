from tqdm import tqdm
from torch.utils.data import DataLoader

# Local Imports
from dataset import LeukemiaDataset
from utils import run_model

# Constants
TEST_IMG_PATH = "data-split/test"

# Hyperparameters
n_epochs = 1
batch_size=None

# Load data
test_dataset = LeukemiaDataset(imgs_path=TEST_IMG_PATH, transforms=None)
test_dataloader = DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=2)

# Testing
run_test = run_model(n_epochs=n_epochs,
                    dataloader=test_dataloader,
                    to_run="test")