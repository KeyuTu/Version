import torch
import torch.nn as nn

class contrast_loss(nn.Module):
    "Contrast_loss"
    def __init__(self, batch_size):
        super(contrast_loss, self).__init__()
        self.batch_size = batch_size
    def forward(self, logits):