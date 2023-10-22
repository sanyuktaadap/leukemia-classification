from tqdm import tqdm

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

# Local imports
from model import LeukemiaClassifier
from metrics import get_metrics

# Constants
CKPT_PATH = "/checkpoints/"
IMG_SIZE = (224, 224)
N_CLASSES = 4

# Hyperparams
lr = None
momentum = None
n_epochs = None

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=IMG_SIZE),
    v2.RandomRotation(degrees=20),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float, scale=True),
])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = LeukemiaClassifier()
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

model = model.to(device)
model.train()

global_step = 0

logger = SummaryWriter()

def run_model(n_epochs=n_epochs,
              dataloader=None,
              to_run=None):
    
    for i in range(n_epochs):
        loss_list = []
        acc_list = []
        sens_list = []
        prec_list = []
        rec_list = []
        f1_list = []

        for x, y in tqdm(dataloader):            

            # Moving input to device
            x = x.to(device)
            y = y.to(device)
            
            # Running forward propagation
            # x (b, c, h, w) -> (b, 4)
            y_hat = model(x)

            loss = loss_fn(y_hat, y)

            if to_run=="train":
                # Make all gradients zero.
                opt.zero_grad()

                # Run backpropagation
                loss.backward()

                # Update parameters
                opt.step()

            loss_list.append(loss.item())
            
            # detach removes y_hat from the original computational graph which might be
            # on gpu.
            y_hat = y_hat.detach().cpu()
            y = y.cpu()
            
            # Compute metrics
            acc = get_metrics(y, y_hat, metric="accuracy")
            acc_list.append(acc)
            
            sens = get_metrics(y, y_hat, metric="sensitivity")
            sens_list.append(sens)
            
            prec = get_metrics(y, y_hat, metrics="precision")
            prec_list.append(prec)
            
            rec = get_metrics(y, y_hat, metrics="recall")
            rec_list.append(rec)   
            
            f1 = get_metrics(y, y_hat, metric="f1")
            f1_list.append(f1)
            
            logger.add_scalar(f"{to_run}/loss", loss.item(), global_step)
            logger.add_scalar(f"{to_run}/accuracy", acc, global_step)
            logger.add_scalar(f"{to_run}/sensitivity", sens, global_step)

            for j in range(N_CLASSES):
                logger.add_scalar(f"{to_run}/precision/{j}", prec[j], global_step)
                logger.add_scalar(f"{to_run}/recall/{j}", rec[j], global_step)
                logger.add_scalar(f"{to_run}/f1/{j}", f1[j], global_step)
        
            global_step += 1
        
        # Print metrics - avg for all
        print(f"Epoch {i} completed - loss, acc, p, r, f1")

        torch.save(model.state_dict(), CKPT_PATH + f"checkpoint{i}.pt")
        
    return print(f'{to_run}, "completed successfully!!!')