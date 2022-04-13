import torch
import torch.nn.functional as F

def compute_kl_loss(mu, logvar):
    loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0)
    return loss


# def compute_rc_loss(y, output):
#     feature_size = y.shape[1]
#     gt = y.reshape(-1,feature_size)
#     out = output.reshape(-1,feature_size)
#
#     loss = F.mse_loss(gt, out, reduction='mean')
#     return loss
def compute_rc_loss(y, output, config):
    feature_size = y.shape[2]
    gt = y.reshape(-1,feature_size)
    out = output.reshape(-1,feature_size)
    if config.rc_loss == "l1":
        loss = F.l1_loss(gt, out, reduction='mean')
    elif config.rc_loss =="l2":
        loss = F.mse_loss(gt, out, reduction='mean')
    else:
        loss = None
    return loss