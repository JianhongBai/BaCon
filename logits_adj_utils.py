import numpy as np
import torch

device = torch.device('cuda:0')

def compute_est_adjustment(tro=1.0, args=None):
    """compute the base probabilities"""

    dist = args.est_dist
    label_freq_array = np.array(dist.cpu())
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    return adjustments