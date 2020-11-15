import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MixcoLoss']


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        
    def forward(self, logits, target):
        probs = F.softmax(logits, 1) 
        nll_loss = (- target * torch.log(probs)).sum(1).mean()

        return nll_loss

class MixcoLoss(nn.Module):
    def __init__(self, mix_param):
        super(MixcoLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.soft_loss = SoftCrossEntropy()
        self.mix_param = mix_param

    def forward(self, outputs):
        if not self.mix_param:
            logits, labels = outputs
            loss = self.loss_fn(logits, labels)
        else:
            logits, labels, logits_mix, lbls_mix = outputs
            loss = self.loss_fn(logits, labels)
            loss += self.mix_param * self.soft_loss(logits_mix, lbls_mix)
        
        return loss  