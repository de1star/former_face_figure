
import torch

import torch.nn.functional as F
import logging
from tqdm import tqdm
import os
import numpy as np
from math import cos, sin

def moving_avg(data, step=5):
    result = np.cumsum(data, axis=0)
    result[step:] = (result[step:] - result[:-step])
    result = result[step - 1:] / step
    return result

def angle2Matrix(x, y, z):
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), sin(x)],
                   [0, -sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, -sin(y)],
                   [0, 1, 0],
                   [sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), sin(z), 0],
                   [-sin(z), cos(z), 0],
                   [0, 0, 1]])
    # rotate
    R = Rx.dot(Ry).dot(Rz)
    R = R.astype(np.float32)
    # print(R)
    return R

def create_pose(rot_x, rot_y, rot_z, trans_x, trans_y, trans_z):
    pose = np.zeros((4, 4))
    pose[:3, :3] = angle2Matrix(rot_x, rot_y, rot_z)
    pose[:,3] = np.asarray([trans_x,trans_y,trans_z,1])
    return pose

def print_network(model, name, log_file):
    """Print out the network information."""
    num_params = sum(p.numel() for p in model.parameters())
    num_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_file.info(name)
    log_file.info(model)
    log_file.info(f"The number of parameters: {num_params_grad}/{num_params}")

def build_tensorboard(log_dir):
    """Build a tensorboard logger."""
    from others.logger import Logger
    return Logger(log_dir)

def create_reg_labels(va_org, device):
    """Generate target va scores for debugging and testing.
    :param va_org: N2
    """
    centers = [
        [0.502170*2-1, 0.497995*2-1],
        [0.832602*2-1, 0.567080*2-1],
        [0.156761*2-1, 0.370705*2-1],
        [0.615316*2-1, 0.840581*2-1],
        [0.442205*2-1, 0.887155*2-1],
        [0.134421*2-1, 0.717098*2-1],
        [0.317687*2-1, 0.830228*2-1],
        [0.222583*2-1, 0.794049*2-1],
    ]
    va_trg_list = []
    for i in range(8):
        va_trg = torch.tensor([centers[i]] * va_org.size(0))
        va_trg_list.append(va_trg.to(device))
    return va_trg_list

def create_lm_labels(va_trg_fixed_list, lm_org_fixed, G_lm):
    lm_trg_fixed_list = []
    for va_trg_fixed in va_trg_fixed_list:
        lm_trg_fixed_list.append(G_lm(lm_org_fixed, va_trg_fixed))

    return lm_trg_fixed_list

def create_reg_labels_grad(va_org, device):
    """Generate target va scores for debugging and testing.
    :param va_org: N2
    """
    centers = [[-1 + 0.2 * x, 1 - 0.2 * y] for x in range(11) for y in range(11)]

    va_trg_list = []
    for i in range(len(centers)):
        va_trg = torch.tensor([centers[i]] * va_org.size(0))
        va_trg_list.append(va_trg.to(device))
    return va_trg_list

def l2_loss(logit, target):
    """Compute mean squared error loss"""
    # return F.mse_loss(logit, target)
    return F.mse_loss(logit, target, size_average=False) / logit.size(0)

def l1_loss(logit, target):
    if logit is None:
        return 0
    else:
        return F.l1_loss(logit, target)

def concordance_cc(r1, r2):
     mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
     CCC = (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)
     return CCC.squeeze()

def ccc_loss(y_hat, y):
    loss = 1 - concordance_cc(y_hat.view(-1), y.view(-1)) #** 2
    return loss

def classifier_loss(logit, target):
    """Compute mean squared error loss"""
    return torch.mean(torch.abs(logit - target))

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def create_neutral_labels(label_org, device):
    c_trg_list = torch.Tensor(0, 2).to(device)
    for arousal, valence in label_org:
        if 0 <= abs(arousal) < 0.1 and 0 <= abs(valence) < 0.1:
            # if the original emotion is neutral, transfer to random emotion
            c_trg_list = torch.cat((c_trg_list, torch.rand(2).unsqueeze_(dim=0).to(device)*2-1))
        else:
            # if the original emotion is random, transfer to neutral
            c_trg_list = torch.cat((c_trg_list, torch.zeros(2).unsqueeze_(dim=0).to(device)*2-1))
    return c_trg_list

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)



def lm_to_imgs(lm, image_size=128, smoother=None):
    #lm.shape=(16,2,68), img.shape=(16,3,128,128), output_lm.shape=(16,68,128,128)
    res = []
    for batch in range(16):
        hot_map = []
        for i in range(68):
            center = (lm.permute(0,2,1)[batch,i]*127).type(torch.long)
            layer = torch.zeros((1,1,image_size,image_size))
            layer[0,0,center[1], center[0]] = 1
            if smoother is not None:
                layer = smoother(layer)
            # layer *= (255 / layer[0,0,center[1], center[0]])
            layer = layer * (1 / layer[0,0,center[1], center[0]])
            hot_map.append(layer)
        hot_map = torch.cat(hot_map,dim=1)
        res.append(hot_map)
    res = torch.cat(res,dim=0)
    return res


#(16,2,68)-->(16,68,128,128)
def lm_to_img(lm, image_size=128, smoother=None):
    #lm.shape=(16,2,68), img.shape=(16,3,128,128), output_lm.shape=(16,68,128,128)
    batch_size = lm.shape[0]
    hot_map = torch.zeros((batch_size,1,image_size,image_size)).to(lm.device)
    lm_p = lm.permute(0,2,1)
    for i in range(68):
        center = (lm_p[:,i]*(image_size-1)).type(torch.long)
        hot_map[torch.arange(batch_size),0, center[:, 1], center[:, 0]] = 1.
    if smoother is not None:
        hot_map = smoother(hot_map)
    # hot_map *= (255 / torch.max(hot_map))
    hot_map *= (1. / torch.max(hot_map))
    return hot_map

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, f'{name}.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger