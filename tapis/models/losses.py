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


def get_weight_from_csv(path, num_classes=None, weight_type="class"):
    """
    Retrieve the weight vector from a csv file.
    
    Args:
        path (str): Path to the CSV file
        num_classes (int): Number of classes
        weight_type (str): "class" for class weights (loss function) or "sample" for sample weights (sampler)
    
    Returns:
        torch.Tensor: Weight tensor
    """
    if path is None or path==False:
        return None
    
    df = pd.read_csv(path)
    if num_classes is not None:
        assert num_classes == len(df), f"Number of classes {num_classes} does not match the number of rows in the csv {len(df)}"
    else:
        num_classes = len(df)
    if 'total_count' not in df.columns:
        # Try with leading/trailing spaces
        count_col = None
        for col in df.columns:
            if 'total_count' in col.strip():
                count_col = col
                break
        if count_col is None:
            print(f"Column 'total_count' not found in csv {path}. Available columns: {df.columns.tolist()}")
            return None
    else:
        count_col = 'total_count'
    
    counts = df[count_col].values
    
    if weight_type == "class":
        # Create class weights by normalizing inverse frequencies
        # This gives higher weight to rare classes
        total_samples = np.sum(counts)
        class_weights = np.zeros(num_classes, dtype=np.float32)
        
        for i, count in enumerate(counts):
            if count > 0:
                # Weight = total_samples / (num_classes * count)
                # This normalizes so that the average weight is 1.0
                class_weights[i] = total_samples / (num_classes * count)
            else:
                class_weights[i] = 0.0
        
        return torch.tensor(class_weights, dtype=torch.float32)
    
    elif weight_type == "sample":
        # Create sample weights by repeating class weights for each sample
        # This is used for WeightedRandomSampler
        total_samples = np.sum(counts)
        sample_weights = []
        
        for i, count in enumerate(counts):
            if count > 0:
                # Weight for each sample of this class
                class_weight = total_samples / (num_classes * count)
                sample_weights.extend([class_weight] * count)
            # If count is 0, no samples for this class
        
        return torch.tensor(sample_weights, dtype=torch.float32)
    
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}. Must be 'class' or 'sample'")