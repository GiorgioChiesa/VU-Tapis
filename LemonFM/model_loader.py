import os
import torch
import torchvision
import torch.nn as nn

def build_LemonFM(nclasses: int = 2, pretrained: bool = True, pretrained_weights = "lemonfm.pth", device=None):


    #net of ConvNext
    net = torchvision.models.convnext_large()
    input_emdim = net.classifier[2].in_features
    net.classifier[2] = nn.Identity()
    
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        state_dict = state_dict['teacher']

        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith('backbone.')}
        msg = net.load_state_dict(state_dict, strict=False)
        print(msg, input_emdim)
        print(f"caricato il checkpoint: ",pretrained_weights)

    if device is not None:
        net.to(device)

    return net
