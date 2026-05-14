#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
from detectron2.layers import ROIAlign
from .build import MODEL_REGISTRY
from torch.nn import functional as F
try:
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

FEATURE_SIZE = {'faster': 1024,
                'mask': 1024,
                'mask_max-mean': 1536,
                'mask_adaptive': 2048,
                'mask_all': 2560,
                'detr': 256,
                'm2f': 512}


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.projection_faster = nn.Linear(256, 1024, bias = True)
        
        self.projection_pathways = nn.Linear(sum(dim_in), 1024, bias = True)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(1024, num_classes, bias=True)
        self.act_func = act_func
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes=None, features=None, **kwargs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection_pathways(x)
        
        if features is not None:
            features = features[:,1:]
            features = self.projection_faster(features)
        x = torch.cat((x, features), axis=1)

        x = self.projection(x)
        
        if self.act_func == "sigmoid" or not self.training:
            x = self.act(x)
        
        return x
    
class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        detach_final_fc=False,
        cfg=None,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            detach_final_fc (bool): if True, detach the fc layer from the
                gradient graph. By doing so, only the final fc layer will be
                trained.
            cfg (struct): The config for the current experiment.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.detach_final_fc = detach_final_fc
        self.cfg = cfg
        self.local_projection_modules = []
        self.predictors = nn.ModuleList()
        self.l2norm_feats = False

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, inputs, **kwargs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x_proj = self.projection(x)

        if not self.training:
            if self.act is not None:
                x_proj = self.act(x_proj)
            # Performs fully convlutional inference.
            if x_proj.ndim == 5 and x_proj.shape[1:4] > torch.Size([1, 1, 1]):
                x_proj = x_proj.mean([1, 2, 3])

        x_proj = x_proj.view(x_proj.shape[0], -1)

        return x_proj


class TransformerBasicHead(nn.Module):
    """
    Frame Classification Head of TAPIS.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False,
        recognition=False
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.class_projection = nn.Linear(dim_in, num_classes, bias=True)
        self.cls_embed = cls_embed
        self.recognition = recognition
        self.act_func = act_func

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, cls_idx=1, **kwargs):
        x = inputs
        if self.cls_embed and not self.recognition:
            x = x[:, cls_idx]
        elif self.cls_embed:
            x = x[:,1:].mean(1)
        else:
            x = x.mean(1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.class_projection(x)

        if self.act_func == "sigmoid" or not self.training:
            x = self.act(x)
        return x


@MODEL_REGISTRY.register()
class BinaryClassificationHead(nn.Module):
    """
    One-vs-all binary classification head.

    Produces independent binary scores for each class.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="sigmoid",
        cls_embed=False,
        recognition=False,
    ):
        super(BinaryClassificationHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.class_projection = nn.Linear(dim_in, num_classes, bias=True)
        self.act_func = act_func
        self.cls_embed = cls_embed
        self.recognition = recognition

        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func in ["none", "logits"]:
            self.act = None
        else:
            raise NotImplementedError(
                f"{act_func} is not supported as an activation function."
            )

    def forward(self, inputs, cls_idx=1, **kwargs):
        x = inputs
        if self.cls_embed and not self.recognition:
            x = x[:, cls_idx]
        elif self.cls_embed:
            x = x[:, 1:].mean(1)
        else:
            x = x.mean(1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.class_projection(x)

        if self.act is not None and (self.act_func == "sigmoid" or not self.training):
            x = self.act(x)
        return x


class ClassificationBasicHead(nn.Module):
    """
    Frame Classification Head of TAPIS.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ClassificationBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.class_projection = nn.Linear(dim_in, num_classes, bias=True)
        self.act_func = act_func

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, **kwargs):
        x = inputs
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.class_projection(x)
        x = self.act(x)
        return x
    
    

@MODEL_REGISTRY.register()
class TransformerRoIHead(nn.Module):
    """
    Region classification head in TAPIS. 
    """

    def __init__(
        self,
        cfg,
        num_classes=0,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False
    ):
        
        super(TransformerRoIHead, self).__init__()
        self.cfg = cfg
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.cls_embed = cls_embed
        self.use_video = cfg.TASKS.USE_VIDEO
        
        # Region features vector dimension 
        dim_features = cfg.FEATURES.DIM_FEATURES
        
        # Use additional linear layers before temporal pooling
        self.use_prev = cfg.MODEL.TIME_MLP and cfg.MODEL.PREV_MLP
        
        if self.use_video:
        
            if cfg.MODEL.DECODER:
                # Transform features to the same dimensions as MViT's output
                self.feat_project = nn.Sequential(nn.Linear(dim_features,
                                                            768,
                                                            bias=True))
                
                # Transformer decoder layer to do self-attention followed by cross-attention
                decoder_layer = nn.TransformerDecoderLayer(768, 
                                                        cfg.MODEL.DECODER_NUM_HEADS, 
                                                        dim_feedforward=cfg.MODEL.DECODER_HID_DIM,
                                                        batch_first=True)
                # Transformer decoder
                self.decoder = nn.TransformerDecoder(decoder_layer, 
                                                    cfg.MODEL.DECODER_NUM_LAYERS)
                dim_out = 768
                
            elif cfg.MODEL.TIME_MLP:
                if self.use_prev:
                    # Linear layers previous to temporal pooling
                    prev_layers = []
                    for i in range(cfg.MODEL.PREV_MLP_LAYERS):
                        prev_layers.append(nn.Linear(cfg.MODEL.PREV_MLP_HID_DIM if i>0 else 768,
                                                    cfg.MODEL.PREV_MLP_HID_DIM if i<cfg.MODEL.PREV_MLP_LAYERS-1 else cfg.MODEL.PREV_MLP_OUT_DIM,
                                                    bias=True))
                        if i<cfg.MODEL.PREV_MLP_LAYERS-1:
                            prev_layers.append(nn.ReLU())
                    self.prev_pool_project = nn.Sequential(*prev_layers)
                
                # Linear layers after temporal pooling
                post_layers = []
                for i in range(cfg.MODEL.POST_MLP_LAYERS):
                    post_layers.append(nn.Linear(cfg.MODEL.POST_MLP_HID_DIM if i>0 else (cfg.MODEL.PREV_MLP_HID_DIM if self.use_prev else 768),
                                                cfg.MODEL.POST_MLP_HID_DIM if i<cfg.MODEL.POST_MLP_LAYERS-1 else cfg.MODEL.POST_MLP_OUT_DIM,
                                                bias=True))
                    if i<cfg.MODEL.POST_MLP_LAYERS-1:
                        post_layers.append(nn.ReLU())
                self.post_pool_project = nn.Sequential(*post_layers)

                # Linear Layers to transform region feature vectors
                feat_layers = []
                for i in range(cfg.MODEL.FEAT_MLP_LAYERS):
                    feat_layers.append(nn.Linear(cfg.MODEL.FEAT_MLP_HID_DIM if i>0 else dim_features,
                                                cfg.MODEL.FEAT_MLP_HID_DIM if i<cfg.MODEL.FEAT_MLP_LAYERS-1 else cfg.MODEL.FEAT_MLP_OUT_DIM,
                                                bias=True))
                    if i<cfg.MODEL.FEAT_MLP_LAYERS-1:
                        feat_layers.append(nn.ReLU())
                self.feat_project = nn.Sequential(*feat_layers)
                
                dim_out = cfg.MODEL.FEAT_MLP_OUT_DIM + cfg.MODEL.POST_MLP_OUT_DIM
                
            else:
                self.mlp = nn.Sequential(nn.Linear(dim_features, 1024, bias=False),
                                        nn.BatchNorm1d(1024))
                dim_out = 1024 + 768
        else:
            num_classes = cfg.TASKS.NUM_CLASSES[0]
            act_func = cfg.TASKS.HEAD_ACT[0]
            dim_out = dim_features
        
        # Final classification layer 
        self.class_projection = nn.Sequential(nn.Linear(dim_out, num_classes, bias=True),)
        
        self.act_func = act_func
        self.use_act = act_func == 'sigmoid' and cfg.TASKS.LOSS_FUNC[0] != 'bce_logit'
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
    
    def forward(self, inputs, features=None, boxes_mask=None, **kwargs):
        boxes_mask = boxes_mask.bool()

        if self.use_video:
            if self.cls_embed:
                inputs = inputs[:, 1:, :]
            
            if self.cfg.MODEL.DECODER:
                features = self.feat_project(features)
                x = self.decoder(features, inputs, tgt_key_padding_mask=~boxes_mask)
                x = x[boxes_mask]
            
            else:
                if self.use_prev:
                    inputs = self.prev_pool_project(inputs)
                    
                x = inputs.mean(1)
                
                if self.cfg.MODEL.TIME_MLP:
                    x = self.post_pool_project(x)

                max_boxes = boxes_mask.shape[-1] 
                
                # Repeat pooled time features to match the batch dimensions of box proposals
                x_boxes = x.unsqueeze(1).repeat(1,max_boxes,1)[boxes_mask] # Use box mask to remove padding
                
                features = features[boxes_mask] # Use box mask to remove padding
                features = self.feat_project(features)
                
                x = torch.cat([x_boxes, features], dim=1)
        else:
            x = features[boxes_mask]

        x = self.class_projection(x)

        # Only apply final activation for validation or for bce loss
        if self.use_act or not self.training:
            x = self.act(x)

        if self.use_video:
            return x
        else:
            return {self.cfg.TASKS.TASKS[0]:x}


@MODEL_REGISTRY.register()
class MixtureOfExpertHead(nn.Module):
    """
    Mixture of Experts (MoE) Classification Head.
    Combines multiple expert networks with a gating network to produce robust predictions.
    Supports both PyTorch-native experts and sklearn ensemble methods (RandomForest, AdaBoost).
    Compatible with TAPIS frame classification tasks.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        num_experts=3,
        expert_dim_hidden=512,
        dropout_rate=0.0,
        act_func="softmax",
        sklearn_method="random_forest",
        sklearn_n_estimators=100,
        cls_embed=False,
        recognition=False,
    ):
        """
        Args:
            dim_in (int): Input feature dimension
            num_classes (int): Number of output classes
            num_experts (int): Number of expert networks
            expert_dim_hidden (int): Hidden dimension for each expert
            dropout_rate (float): Dropout probability
            act_func (str): Activation function ('softmax' or 'sigmoid')
            use_sklearn (bool): Use sklearn ensemble for experts
            sklearn_method (str): 'random_forest' or 'adaboost'
            sklearn_n_estimators (int): Number of estimators for sklearn
            cls_embed (bool): Whether to use class embedding (for compatibility)
            recognition (bool): Whether this is for presence recognition task
        """
        super(MixtureOfExpertHead, self).__init__()
        
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.sklearn_method = sklearn_method
        self.use_sklearn = HAS_SKLEARN and sklearn_method in ["random_forest", "adaboost"]
        self.act_func = act_func
        self.cls_embed = cls_embed
        self.recognition = recognition
        
        # Experts: Small neural networks
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(dim_in, expert_dim_hidden, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
                nn.Linear(expert_dim_hidden, num_classes, bias=True)
            )
            self.experts.append(expert)
        
        # Gating network: Learns to weight expert outputs
        self.gating_network = nn.Sequential(
            nn.Linear(dim_in, expert_dim_hidden, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(expert_dim_hidden, num_experts, bias=True),
            nn.Softmax(dim=1)  # Weight distribution across experts
        )
        
        # Final projection (optional)
        self.final_proj = nn.Linear(num_classes, num_classes, bias=True)
        
        # Activation function
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                f"{act_func} is not supported as an activation function."
            )
        
        # sklearn experts (optional, for inference)
        self.sklearn_experts = None
        self.sklearn_method = sklearn_method
        self.sklearn_n_estimators = sklearn_n_estimators
        self.is_trained = False

    def fit_sklearn_experts(self, X, y):
        """
        Train sklearn ensemble experts on data. Useful for transfer learning.
        
        Args:
            X (array-like): Training features of shape (n_samples, dim_in)
            y (array-like): Training labels of shape (n_samples,)
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for sklearn experts. Install with: pip install scikit-learn")
        
        self.sklearn_experts = []
        
        for _ in range(self.num_experts):
            if self.sklearn_method == "random_forest":
                expert = RandomForestClassifier(
                    n_estimators=self.sklearn_n_estimators,
                    max_depth=15,
                    random_state=None
                )
            elif self.sklearn_method == "adaboost":
                expert = AdaBoostClassifier(
                    n_estimators=self.sklearn_n_estimators,
                    random_state=None
                )
            else:
                raise ValueError(f"sklearn_method must be 'random_forest' or 'adaboost', got {self.sklearn_method}")
            
            expert.fit(X, y)
            self.sklearn_experts.append(expert)
        
        self.is_trained = True

    def forward(self, inputs, cls_idx=1, use_sklearn_forward=True, **kwargs):
        """
        Forward pass through MoE head.
        
        Args:
            inputs (torch.Tensor): Input features of shape (batch_size, seq_len, dim_in) or (batch_size, dim_in)
            cls_idx (int): Index of cls_embed to use if cls_embed is True
            use_sklearn_forward (bool): Use sklearn experts if available
            **kwargs: Additional arguments (unused)
        
        Returns:
            torch.Tensor: Class predictions of shape (batch_size, num_classes)
        """
        # Handle cls_embed similar to TransformerBasicHead
        x = inputs
        if self.cls_embed and not self.recognition:
            # Extract cls_idx token: (batch_size,)
            x = x[:, cls_idx]
        elif self.cls_embed:
            # For recognition: average all tokens except cls token
            x = x[:, 1:].mean(1)
        else:
            # Default: average all tokens
            x = x.mean(1)
        
        # Now x is (batch_size, dim_in)
        batch_size = x.shape[0]
        
        # Use sklearn experts if requested and trained
        if use_sklearn_forward and self.sklearn_experts is not None and self.is_trained:
            return self._forward_sklearn(x)
        
        # PyTorch experts forward
        # Compute gating weights
        gate_logits = self.gating_network(x)  # (batch_size, num_experts)
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (batch_size, num_classes)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs: (batch_size, num_experts, num_classes)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Expand gate weights for broadcasting: (batch_size, num_experts, 1)
        gate_weights = gate_logits.unsqueeze(2)
        
        # Weighted combination of expert outputs
        # (batch_size, num_experts, num_classes) * (batch_size, num_experts, 1) -> (batch_size, num_experts, num_classes)
        weighted_experts = expert_outputs * gate_weights
        
        # Sum across experts: (batch_size, num_classes)
        moe_output = weighted_experts.sum(dim=1)
        
        # Final projection
        x_out = self.final_proj(moe_output)
        
        # Apply activation
        if self.act_func == "sigmoid" or not self.training:
            x_out = self.act(x_out)
        
        return x_out

    def _forward_sklearn(self, x):
        """
        Forward pass using sklearn experts. Converts to numpy, computes predictions,
        and converts back to torch tensor.
        
        Args:
            x (torch.Tensor): Input features
        
        Returns:
            torch.Tensor: Combined predictions
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        
        # Get predictions from each sklearn expert
        expert_probs = []
        for expert in self.sklearn_experts:
            probs = expert.predict_proba(x_np)  # (batch_size, num_classes)
            expert_probs.append(torch.from_numpy(probs).to(device).to(dtype))
        
        # Stack expert predictions: (batch_size, num_experts, num_classes)
        expert_probs = torch.stack(expert_probs, dim=1)
        
        # Simple averaging across experts (unweighted)
        moe_output = expert_probs.mean(dim=1)  # (batch_size, num_classes)
        
        # Ensure softmax
        moe_output = torch.softmax(moe_output, dim=1)
        
        return moe_output


@MODEL_REGISTRY.register()
class SklearnEnsembleHead(nn.Module):
    """
    Pure Sklearn Ensemble Head wrapper for PyTorch.
    Uses trained sklearn RandomForest or AdaBoost for classification.
    Suitable for inference and knowledge distillation.
    """

    def __init__(
        self,
        num_classes,
        ensemble_method="random_forest",
        n_estimators=100,
    ):
        """
        Args:
            num_classes (int): Number of output classes
            ensemble_method (str): 'random_forest' or 'adaboost'
            n_estimators (int): Number of estimators
        """
        super(SklearnEnsembleHead, self).__init__()
        
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        self.n_estimators = n_estimators
        self.ensemble_model = None
        self.is_trained = False
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    def fit(self, X, y):
        """
        Train the ensemble classifier.
        
        Args:
            X (array-like): Training features
            y (array-like): Training labels
        """
        if self.ensemble_method == "random_forest":
            self.ensemble_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=20,
                random_state=42
            )
        elif self.ensemble_method == "adaboost":
            self.ensemble_model = AdaBoostClassifier(
                n_estimators=self.n_estimators,
                random_state=42
            )
        else:
            raise ValueError(f"ensemble_method must be 'random_forest' or 'adaboost'")
        
        self.ensemble_model.fit(X, y)
        self.is_trained = True

    def forward(self, x, **kwargs):
        """
        Forward pass (inference only).
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, feature_dim)
        
        Returns:
            torch.Tensor: Class probabilities of shape (batch_size, num_classes)
        """
        if self.ensemble_model is None or not self.is_trained:
            raise RuntimeError("Ensemble model must be trained before forward pass. Call .fit() first.")
        
        device = x.device
        dtype = x.dtype
        
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        
        # Get predictions
        probs = self.ensemble_model.predict_proba(x_np)
        
        # Convert back to torch
        probs_torch = torch.from_numpy(probs).to(device).to(dtype)
        
        return probs_torch
    
    
    
    
    
@MODEL_REGISTRY.register()
class CascadeClassificationHead(nn.Module):
    """
    Two-stage Cascade Classification Head.
    
    Stage 1: Binary classifier distinguishes between Idle (0) and Event (1-N)
    Stage 2: Multi-class classifier for event types (only applied when event is non-idle)
    
    Supports both end-to-end and two-phase training modes.
    Label format: 0 = Idle, 1-num_event_classes = Event types
    """
    
    def __init__(
        self,
        dim_in,
        num_event_classes,
        stage1_hidden_dim=512,
        stage2_hidden_dim=512,
        dropout_rate=0.0,
        act_func="softmax",
        stage1_weight=1.0,
        stage2_weight=0.5,
        training_mode="end-to-end",
        cls_embed=False,
        recognition=False,
    ):
        """
        Args:
            dim_in (int): Input feature dimension
            num_event_classes (int): Number of event types (e.g., 33 for steps)
            stage1_hidden_dim (int): Hidden dimension for Stage 1 (binary classifier)
            stage2_hidden_dim (int): Hidden dimension for Stage 2 (event classifier)
            dropout_rate (float): Dropout probability
            act_func (str): Activation function ('softmax' or 'sigmoid')
            stage1_weight (float): Loss weight for Stage 1 (default: 1.0)
            stage2_weight (float): Loss weight for Stage 2 (default: 0.5)
            training_mode (str): 'end-to-end' or 'two-phase'
            cls_embed (bool): Whether to use class embedding
            recognition (bool): Presence recognition mode
        """
        super(CascadeClassificationHead, self).__init__()
        
        self.dim_in = dim_in
        self.num_event_classes = num_event_classes
        self.num_output_classes = num_event_classes + 1  # +1 for Idle
        self.act_func = act_func
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight
        self.training_mode = training_mode
        self.cls_embed = cls_embed
        self.recognition = recognition
        self.current_training_phase = 1  # 1 or 2 for two-phase training
        
        # Stage 1: Binary Classifier (Idle vs Event)
        self.stage1 = nn.Sequential(
            nn.Linear(dim_in, stage1_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(stage1_hidden_dim, 2, bias=True),  # Binary: Idle / Event
        )
        
        # Stage 2: Event Classifier (33 event types)
        self.stage2 = nn.Sequential(
            nn.Linear(dim_in, stage2_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(stage2_hidden_dim, num_event_classes, bias=True),
        )
        
        # Activation functions
        if act_func == "softmax":
            self.act1 = nn.Softmax(dim=1)
            self.act2 = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act1 = nn.Sigmoid()
            self.act2 = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{act_func} not supported")
    
    def set_training_phase(self, phase):
        """
        Set training phase for two-phase training mode.
        
        Phase 1: Train only Stage 1 (binary classifier)
        Phase 2: Train only Stage 2 (event classifier, Stage 1 frozen)
        
        Args:
            phase (int): 1 or 2
        """
        if phase not in [1, 2]:
            raise ValueError("phase must be 1 or 2")
        
        self.current_training_phase = phase
        
        if phase == 1:
            # Freeze Stage 2, enable Stage 1
            for param in self.stage2.parameters():
                param.requires_grad = False
            for param in self.stage1.parameters():
                param.requires_grad = True
        else:  # phase == 2
            # Freeze Stage 1, enable Stage 2
            for param in self.stage1.parameters():
                param.requires_grad = False
            for param in self.stage2.parameters():
                param.requires_grad = True
    
    def forward(self, inputs, cls_idx=1, labels=None, **kwargs):
        """
        Forward pass through cascade head.
        
        Args:
            inputs (torch.Tensor): Input features (batch, seq_len, dim_in) or (batch, dim_in)
            cls_idx (int): CLS token index if cls_embed is True
            labels (torch.Tensor): Ground truth labels (batch,) for training
                                  0 = Idle, 1-num_event_classes = Event types
            **kwargs: Additional arguments
        
        Returns:
            torch.Tensor: Combined logits (batch, num_event_classes + 1)
                         or dict with loss if labels provided
        """
        # Handle cls_embed similar to TransformerBasicHead
        x = inputs
        if self.cls_embed and not self.recognition:
            x = x[:, cls_idx]
        elif self.cls_embed:
            x = x[:, 1:].mean(1)
        else:
            x = x.mean(1)
        
        # Now x is (batch, dim_in)
        batch_size = x.shape[0]
        device = x.device
        
        # Stage 1: Binary classification (Idle vs Event)
        stage1_logits = self.stage1(x)  # (batch, 2)
        
        # Stage 2: Event classification (33 types)
        stage2_logits = self.stage2(x)  # (batch, num_event_classes)
        
        # Combine outputs
        if self.training and labels is not None:
            # Training mode: compute losses
            if self.training_mode == "end-to-end":
                loss = self._compute_loss_end_to_end(stage1_logits, stage2_logits, labels)
            else:  # two-phase
                loss = self._compute_loss_two_phase(stage1_logits, stage2_logits, labels)
            
            return {"logits": self._combine_outputs(stage1_logits, stage2_logits), 
                    "loss": loss}
        else:
            # Inference mode: return combined logits
            return self._combine_outputs(stage1_logits, stage2_logits)
    
    def _combine_outputs(self, stage1_logits, stage2_logits):
        """
        Combine Stage 1 (binary) and Stage 2 (event) outputs.
        
        Returns logits for all classes: P(Idle) and P(Event_i) for i in 1..num_event_classes
        
        Args:
            stage1_logits (torch.Tensor): (batch, 2) - [idle_logit, event_logit]
            stage2_logits (torch.Tensor): (batch, num_event_classes)
        
        Returns:
            torch.Tensor: (batch, num_event_classes + 1) combined logits
        """
        batch_size = stage1_logits.shape[0]
        device = stage1_logits.device
        
        # Get probabilities
        stage1_probs = F.softmax(stage1_logits, dim=1)  # (batch, 2)
        stage2_probs = F.softmax(stage2_logits, dim=1)  # (batch, num_event_classes)
        
        # P(Idle) is probability of first bin in Stage 1
        p_idle = stage1_probs[:, 0]  # (batch,)
        
        # P(Event) is probability of second bin in Stage 1
        p_event = stage1_probs[:, 1]  # (batch,)
        
        # Combined probabilities: P(Idle) and P(Event_i) = P(Event) * P(Event_i | Event)
        combined_probs = torch.zeros(batch_size, self.num_output_classes, 
                                     device=device, dtype=stage1_probs.dtype)
        
        combined_probs[:, 0] = p_idle  # P(Idle)
        combined_probs[:, 1:] = p_event.unsqueeze(1) * stage2_probs  # P(Event) * P(Event_i|Event)
        
        # Convert back to logits for consistency
        combined_logits = torch.log(combined_probs + 1e-8)
        
        return combined_logits
    
    def _compute_loss_end_to_end(self, stage1_logits, stage2_logits, labels):
        """
        Compute loss for end-to-end training.
        
        Both stages see gradients and can improve each other.
        - Stage 1: Learn to distinguish Idle (0) vs Event (1-33)
        - Stage 2: Learn to classify event types
        
        Args:
            stage1_logits (torch.Tensor): (batch, 2)
            stage2_logits (torch.Tensor): (batch, num_event_classes)
            labels (torch.Tensor): (batch,) ground truth labels
        
        Returns:
            torch.Tensor: scalar loss
        """
        # Stage 1 loss: Binary classification
        # Convert labels: 0 → 0 (Idle), 1-33 → 1 (Event)
        binary_labels = (labels > 0).long()  # 0 for Idle, 1 for Event
        
        loss_stage1 = F.cross_entropy(stage1_logits, binary_labels, reduction="mean")
        
        # Stage 2 loss: Event classification (only for non-idle samples)
        # For Idle samples (labels == 0), Stage 2 loss doesn't apply
        event_mask = labels > 0  # Boolean mask for event samples
        
        if event_mask.any():
            # Adjust labels for Stage 2: subtract 1 (since events are 1-33)
            event_labels = labels[event_mask] - 1  # Convert to 0-32
            event_logits = stage2_logits[event_mask]  # Select event samples
            
            loss_stage2 = F.cross_entropy(event_logits, event_labels, reduction="mean")
        else:
            loss_stage2 = torch.tensor(0.0, device=stage1_logits.device, 
                                      dtype=stage1_logits.dtype)
        
        # Combined loss
        total_loss = (self.stage1_weight * loss_stage1 + 
                     self.stage2_weight * loss_stage2)
        
        return total_loss
    
    def _compute_loss_two_phase(self, stage1_logits, stage2_logits, labels):
        """
        Compute loss for two-phase training.
        
        Phase 1: Only train Stage 1 (binary classifier)
        Phase 2: Only train Stage 2 (event classifier)
        
        Args:
            stage1_logits (torch.Tensor): (batch, 2)
            stage2_logits (torch.Tensor): (batch, num_event_classes)
            labels (torch.Tensor): (batch,) ground truth labels
        
        Returns:
            torch.Tensor: scalar loss
        """
        if self.current_training_phase == 1:
            # Phase 1: Train binary classifier
            binary_labels = (labels > 0).long()
            loss = F.cross_entropy(stage1_logits, binary_labels, reduction="mean")
        
        else:  # phase == 2
            # Phase 2: Train event classifier
            event_mask = labels > 0
            
            if event_mask.any():
                event_labels = labels[event_mask] - 1
                event_logits = stage2_logits[event_mask]
                loss = F.cross_entropy(event_logits, event_labels, reduction="mean")
            else:
                # No events in batch, return zero loss
                loss = torch.tensor(0.0, device=stage2_logits.device,
                                   dtype=stage2_logits.dtype)
        
        return loss


