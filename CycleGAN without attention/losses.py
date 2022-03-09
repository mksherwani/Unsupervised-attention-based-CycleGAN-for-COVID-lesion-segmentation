"""
Copyright (c) 2018 Maria Francesca Spadea.
- All Rights Reserved -

Unauthorized copying/distributing/editing/using/selling of this file (also partial), via any medium, is strictly prohibited.

The code is proprietary and confidential.

The software is just for research purpose, it is not intended for clinical application or for use in which the failure of the software
could lead to death, personal injury, or severe physical or environmental damage.
"""

import torch
import torch.nn as nn
import torch.functional as f

from math_utils import batch_fftshift2d
import pytorch_ssim

import pdb

class GlobalMAE(nn.Module):

    def __init__(self):
        super(GlobalMAE, self).__init__()

    def forward(self, gt, comp):
        return torch.mean(torch.abs(gt-comp))

class MaskedMAE(nn.Module):

    def __init__(self):
        super(MaskedMAE, self).__init__()

    def forward(self, gt, comp, skin, weight=1.0):
        return weight * torch.sum(torch.abs(skin * (gt - comp)))/torch.sum(skin)

class MaskedWeightedMAE(nn.Module):

    def __init__(self):
        super(MaskedWeightedMAE, self).__init__()

    def forward(self, gt, comp, skin, mae_weight=1.0):
        abs_diff = torch.abs(gt - comp)
        #w = abs_diff/2400.0
        ##w = torch.ones_like(abs_diff)
        ##w[abs_diff<60]=0.5
        #w = skin.clone()
        #w[abs_diff<=50]=0
        w = torch.zeros_like(skin)
        w[gt>=200]=1

        weighted_mae = torch.sum(skin * w * abs_diff)/torch.sum(skin*w)

        return mae_weight * weighted_mae

class MaskedMAEPlusFFT(nn.Module):

    def __init__(self):
        super(MaskedMAEPlusFFT, self).__init__()

    def forward(self, gt, comp, skin, fft_weight):
        maskedMAE = torch.sum(torch.abs(skin * (gt - comp)))/torch.sum(skin)
        
        comp[skin==0]=-1000
        gt_fft = batch_fftshift2d(torch.rfft(gt, 2, normalized=True, onesided=False))
        comp_fft = batch_fftshift2d(torch.rfft(comp, 2, normalized=True, onesided=False))

        diff_fft = torch.mean(torch.abs(gt_fft - comp_fft))

        fft_total = fft_weight * diff_fft
        #pdb.set_trace()
        print("MAE = %.2f -- FFT = %.2f" % (float(maskedMAE), float(fft_total)))
        #return maskedMAE + fft_total
        return fft_total

        #return torch.add(maskedMAE, fft_weight, diff_fft)

class MaskedMAEPlusSSIM(nn.Module):

    def __init__(self):
        super(MaskedMAEPlusSSIM, self).__init__()

    def forward(self, gt, comp, skin, ssim_weight):
        maskedMAE = torch.sum(torch.abs(skin * (gt - comp)))/torch.sum(skin)
        
        comp[skin==0]=-1000

        ssim_losser = pytorch_ssim.SSIM(window_size = 11)
        ssim_loss = (1.0 - ssim_losser(gt, comp))
        total_ssim_loss = ssim_loss * ssim_weight 

        #pdb.set_trace()
        print("MAE = %.2f -- SSIM = %.2f -- TOTAL SSIM = %.2f" % (float(maskedMAE), float(ssim_loss), float(total_ssim_loss)))
        
        return maskedMAE + total_ssim_loss

