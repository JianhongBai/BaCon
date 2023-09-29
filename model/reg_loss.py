import torch
import numpy as np
import math
import torch.nn.functional as F


def compute_reg_loss(student_out, est_count, args):

    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
    avg_probs = avg_probs * 1 / (est_count) ** args.p
    avg_probs /= avg_probs.sum()
    me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
    return me_max_loss
