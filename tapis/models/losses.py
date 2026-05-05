#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd
import numpy as np


# class FocalLoss(nn.Module):
#     """
#     Focal Loss for addressing class imbalance.
#     """
#     def __init__(self, weight=None, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.weight = weight
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
        
    # def forward(self, inputs, targets):
    #     if self.weight is not None:
    #         ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
    #     else:
    #         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    #     pt = torch.exp(-ce_loss)
    #     focal_loss = ((1 - pt) ** self.gamma) * ce_loss
    #     if self.alpha is not None:
    #         if isinstance(self.alpha, (list, np.ndarray)):
    #             alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
    #         elif isinstance(self.alpha, torch.Tensor):
    #             alpha_t = self.alpha[targets]
    #         else:
    #             alpha_t = self.alpha
    #         focal_loss = alpha_t * focal_loss
    #     if self.reduction == 'mean':
    #         return focal_loss.mean()
    #     elif self.reduction == 'sum':
    #         return focal_loss.sum()
    #     else:
    #         return focal_loss
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, alpha=None, reduction='mean'):
        """
        gamma: focusing parameter (più alto = più focus sugli esempi difficili)
        alpha: tensor di pesi per classe (shape: [num_classes]) oppure None
        reduction: 'mean', 'sum' o 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = weight if weight is not None else alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size] (class indices)
        """

        # log softmax per stabilità numerica
        log_probs = F.log_softmax(c, dim=1)
        probs = torch.exp(log_probs)

        # seleziona la probabilità della classe corretta
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # focal term
        focal_term = (1 - pt) ** self.gamma

        # alpha weighting (se fornito)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        # reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss    


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "focal_loss": FocalLoss,
}

_TYPES = {
    "cross_entropy": torch.long,
    "bce": torch.float,
    "bce_logit": torch.float,
    "focal_loss": torch.long,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

def get_loss_type(loss_name,presicion):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _TYPES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    if presicion==64 and _TYPES[loss_name]==torch.float:
        return torch.double
    return _TYPES[loss_name]

def compute_weighted_loss(losses, weight_vector):
    """
    Weighted loss function
    """
    final_loss = 0
    for ind, loss in enumerate(losses):
        final_loss+= loss * weight_vector[ind]
    return final_loss


def get_weight_from_csv(path, num_classes=None):
    """
    Retrieve the weight vector from a csv file.
    Args (str):
    """
    if path is None or path==False:
        return None
    
    df = pd.read_csv(path)
    if num_classes is not None:
        assert num_classes == len(df), f"Number of classes {num_classes} does not match the number of rows in the csv {len(df)}"
    else:
        num_classes = len(df)
    if 'total_count' not in df.columns:
        print(f"Column 'total_count' not found in csv {path}. Please make sure the csv has a column named 'total_count' with the count of samples for each class.")
        return None
    counts = df['total_count'].values
    inverted = [1.0 / val if val != 0 else 0.0 for val in counts]
    # inverted[:3] = [val * 0.01 for val in inverted[:3]]  # set the first three classes to 0 weight
    return torch.tensor(inverted, dtype=torch.float32)