from tqdm import tqdm
import torch
# Torch imports
from metrics import get_metrics

def run_epoch(dataloader,
              model, 
              device, 
              loss_fn,
              logger, 
              opt=None, 
              to_run="train", 
              n_classes=4, 
              step=0):
    """_summary_

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        device (_type_): _description_
        loss_fn (_type_): _description_
        logger (_type_): _description_
        opt (_type_, optional): _description_. Defaults to None.
        to_run (str, optional): _description_. Defaults to "train".
        n_classes (int, optional): _description_. Defaults to 4.
        step (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    
    loss_list = []
    acc_list = []
    spec_list = []
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

        if to_run == "train":
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
        acc = get_metrics(y_hat, y, metric="accuracy")
        acc_list.append(acc)

        spec = get_metrics(y_hat, y, metric="specificity")
        spec_list.append(spec)

        prec = get_metrics(y_hat, y, metric="precision")
        prec_list.append(prec)

        rec = get_metrics(y_hat, y, metric="recall")
        rec_list.append(rec)

        f1 = get_metrics(y_hat, y, metric="f1")
        f1_list.append(f1)

        logger.add_scalar(f"{to_run}/loss", loss.item(), step)
        logger.add_scalar(f"{to_run}/accuracy", acc, step)

        for j in range(n_classes):
            logger.add_scalar(f"{to_run}/precision/{j}", prec[j], step)
            logger.add_scalar(f"{to_run}/recall/{j}", rec[j], step)
            logger.add_scalar(f"{to_run}/f1/{j}", f1[j], step)
            logger.add_scalar(f"{to_run}/specificity/{j}",
                              spec[j], step)

        step += 1

    avg_loss = torch.Tensor(loss_list).mean()
    avg_acc = torch.Tensor(acc_list).mean()
    avg_spec = torch.vstack(spec_list).mean()
    avg_p = torch.vstack(prec_list).mean()
    avg_r = torch.vstack(rec_list).mean()
    avg_f1 = torch.vstack(f1_list).mean()
        
    return avg_loss, avg_acc, avg_spec, avg_p, avg_r, avg_f1