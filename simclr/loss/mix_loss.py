import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, logits, target):
        probs = F.softmax(logits, 1) 
        loss = (- target * torch.log(probs)).sum(1).mean()

        return loss
