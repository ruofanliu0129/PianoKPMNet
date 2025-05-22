import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss 



def cal_ot_loss(gt_sequence, pred_sequence, blur=0.05):
    assert pred_sequence.shape == gt_sequence.shape, "Shape mismatch between pred and gt"
    sinkhorn = SamplesLoss("sinkhorn", blur=blur).to(pred_sequence.device)
    ot_loss = sinkhorn(pred_sequence, gt_sequence)

    return ot_loss.item()



def cal_mse_loss(gt_sequence, pred_sequence):
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(gt_sequence, pred_sequence)

    return mse_loss.item()



class EmgLoss(nn.Module):

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'mse': 1.0,
            'ot': 1.0
        }
        self.sinkhorn = SamplesLoss("sinkhorn", blur=0.05)

    def forward(self, pred_sequence, gt_sequence):
        """
        Args:
            pred_sequence: (B, T, C)
            gt_sequence: (B, T, C)
        Returns:
            total_loss: scalar
            loss_dict: dictionary containing individual losses
        """
        B, T, C = pred_sequence.shape
        assert pred_sequence.shape == gt_sequence.shape, "Shape mismatch between pred and gt"
        
        # MSE loss
        mse_loss = torch.mean((pred_sequence - gt_sequence) ** 2)

        # OT loss
        ot_losses = []
        for b in range(B):
            pred_seq_b = pred_sequence[b]  # (T, C)
            gt_seq_b = gt_sequence[b]    # (T, C)
            
            ot_loss_b = self.sinkhorn(
                pred_seq_b,  # (T, C)
                gt_seq_b     # (T, C)
            )
            ot_losses.append(ot_loss_b)
        
        ot_loss = torch.mean(torch.stack(ot_losses))

        total_loss = (
            self.weights['mse'] * mse_loss +
            self.weights['ot'] * ot_loss
        )

        loss_dict = {
            'mse': mse_loss,
            'ot': ot_loss
        }

        return total_loss, loss_dict

