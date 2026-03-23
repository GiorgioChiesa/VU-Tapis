#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd
import numpy as np


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    
}

_TYPES = {
    "cross_entropy": torch.long,
    "bce": torch.float,
    "bce_logit": torch.float,
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
    inverted[:3] = [val * 0.01 for val in inverted[:3]]  # set the first three classes to 0 weight
    return torch.tensor(inverted / np.sum(inverted), dtype=torch.float32)