import glob
import os
from PIL import Image
from torch.utils.data import Dataset

class LeukemiaDataset(Dataset):
    """Leukemia Dataset
    Args:
        Dataset : Parent torch dataset class
    """
    def __init__(
        self, 
        imgs_path, 
        img_dim,
        transforms=None,
        class_map={"benign" : 0, "early": 1, "pre": 2, "pro": 3},
    ):
        """Initializes a leukemia dataset instance
        Args:
            imgs_path (str): Path to the folder containing classwise images
            img_dim (tuple): Tuple containing final image height and width
            transforms (torchvision.transforms): Transforms to be run on an image
            class_map (dict, optional): # Provides the mapping from the name to the number 
                Defaults to {"Benign" : 0, "Early": 1, "Pre": 2, "Pro": 3}.
        """
        
        # Path to training data
        self.imgs_path = imgs_path
        self.class_map = class_map
        self.img_dim = img_dim
        self.transforms = transforms
        # List containing paths to all the images
        self.data = []
        # List of all folders inside imgs_path
        file_list = glob.glob(os.path.join(self.imgs_path, "*"))
        # Iterate over all the classes in file list
        for class_path in file_list:
            # For each class, extract the actual class name
            class_path = class_path.rstrip('/\\')
            class_name = class_path.split('\\')[-1].lower()
            # Retrieve each image (.jpg) in class folders
            for img_path in glob.glob(os.path.join(class_path, "*")):
                # self.data will contain path to each image, the respective class
                self.data.append((img_path, class_name))
        
    def __len__(self):
        """Gets the length of the dataset

        Returns:
            int: total number of data points
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """Gets the indexed items from the dataset
        Args:
            idx (int): index number
        Returns:
            vector, int: indexed image with its corresponding label
        """
        # idx - indexing data for accessibility 
        img_path, class_name = self.data[idx]
        # Assigning ids to each class (number, not name of the class)
        class_id = self.class_map[class_name]
        # Loads an image from the given image_path
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
            
        return img, class_id