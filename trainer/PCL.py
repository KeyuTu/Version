"""
The Implementation is based on Pytorch
"""

from cmath import log
import torch
import torch.nn.functional as F
from loss import builder as loss_builder
from loss.soft_supconloss import SoftSupConLoss

from .base_trainer import Trainer

class PCL(Trainer):
    
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super().__init__(cfg=cfg)

        self.all_cfg = all_cfg
        self.device = device
        if self.cfg.amp:
            from apex import amp
            self.amp = amp
        self.loss_x = loss_builder.build(cfg.loss_x)
        self.loss_u = loss_builder.build(cfg.loss_u)
        self.loss_contrast = SoftSupConLoss(temperature=self.cfg.temperature)

        self.pseudo_with_ema = False
        self.da = False
        self._get_config()
    
    def _get_config(self):
        if self.all_cfg.get("ema", False):
            self.pseudo_with_ema = self.all_cfg.ema.get(
                "``pseudo_with_ema``", False)

        # distribution alignment mentioned in paper
        self.prob_list = []
        if self.cfg.get("DA", False):
            self.da = self.cfg.DA.use

    def _da_pseudo_label(self, prob_list, logits_u_w):
        """ distribution alignment
        """
        with torch.no_grad():
            probs = torch.softmax(logits_u_w, dim=1)

            prob_list.append(probs.mean(0))
            if len(prob_list) > self.cfg.DA.da_len:
                prob_list.pop(0)
            prob_avg = torch.stack(prob_list, dim=0).mean(0)
            # print(prob_avg.shape)
            # exit()
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            probs = probs.detach()
        return probs
    
    def compute_loss(self,
                     data_x,
                     data_u,
                     model,
                     optimizer,
                     ema_model=None,
                     **kwargs):
        # make inputs
        inputs_x, targets_x = data_x
        inputs_x = inputs_x[0]

        inputs_u, targets_u = data_u
        inputs_u_w, inputs_u_s, inputs_u_s1 = inputs_u

        batch_size = inputs_x.shape[0]
        targets_x = targets_x.to(self.device)

        if self.pseudo_with_ema:
            None
        else:
            inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1],
                                dim=0).to(self.device)
            logits, features = model(inputs)
            print(features.shape)
            exit()
            
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s, _ = logits[batch_size:].chunk(3)
            _, f_u_s, f_u_s1 = features[batch_size:].chunk(3)
            del logits
            del features
            del _

        Lx = self.loss_x(logits_x, targets_x, reduction='mean')
        if not self.da:
            prob_u_w = torch.softmax(logits_u_w.detach() / self.cfg.T, dim=-1)
        else:
            prob_u_w = self._da_pseudo_label(logits_u_w)

        max_probs, p_targets_u = torch.max(prob_u_w, dim=-1)
        mask = max_probs.ge(self.cfg.threshold).float()
        Lu = (self.loss_u(logits_u_s, p_targets_u, reduction='none') *
                mask).mean()
        
        labels = p_targets_u
        features = torch.cat([f_u_s.unsqueeze(1), f_u_s1.unsqueeze(1)], dim=1)
        
        Lcontrast = self.loss_contrast(features, max_probs=max_probs, labels=labels)

        loss = Lx + self.cfg.lambda_u * Lu + self.cfg.lambda_contrast * Lcontrast

        if hasattr(self, "amp"):
            with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif "SCALER" in kwargs and kwargs["SCALER"] is not None:
            kwargs['SCALER'].scale(loss).backward()
        else:
            # entrance
            loss.backward()

        targets_u = targets_u.to(self.device)
        right_labels = (p_targets_u == targets_u).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

        loss_dict = {
            "loss": loss,
            "loss_x": Lx,
            "loss_u": Lu,
            "loss_contrast": Lcontrast,
            "mask_prob": mask.mean(),
            "pseudo_acc": pseudo_label_acc,
        }

        return loss_dict


