import imp
from tracemalloc import start
import torch
import torch.nn.functional as F
import torch.nn as nn

class contrast_loss(nn.Module):
    "Contrast_loss"
    def __init__(self, batch_size, mu):
        super(contrast_loss, self).__init__()
        self.batch_size = batch_size
        self.mu = mu
    
    def forward(self, logits):
        device = (torch.device('cuda') if logits.is_cuda 
                  else torch.device('cpu'))
        criterion = nn.CrossEntropyLoss().to(device)
        
        probs = F.softmax(logits)
        sim_matrix = torch.mm(probs, probs.transpose(0, 1).contiguous())
        shield_matrix = torch.eye(2*self.batch_size*self.mu).to(device)

        sim_matrix = sim_matrix * (1-shield_matrix) - (10**5) * shield_matrix
        target = torch.arange(start=0, end=self.mu*self.batch_size, step=1).to(device)
        lb = torch.cat([target+self.batch_size*self.mu, target])
         
        loss = criterion(probs, lb)
 
        return loss

        



