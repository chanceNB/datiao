"""Stride-one residual TCN with left-only convolutions."""
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,dilation=dilation,padding=0)
        self.left_padding = (kernel_size-1)*dilation

    def forward(self, x):
        return self.conv(F.pad(x,(self.left_padding,0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(CausalConv1d(in_channels,out_channels,kernel_size,dilation),nn.ReLU(),nn.Dropout(dropout),CausalConv1d(out_channels,out_channels,kernel_size,dilation),nn.ReLU(),nn.Dropout(dropout))
        self.residual = nn.Identity() if in_channels==out_channels else nn.Conv1d(in_channels,out_channels,1)

    def forward(self, x):
        return self.layers(x) + self.residual(x)


def compute_receptive_field(kernel_size, dilations, convs_per_block=2):
    return 1 + sum((kernel_size-1)*d*convs_per_block for d in dilations)


class PenTCN(nn.Module):
    def __init__(self, input_channels=24, hidden_channels=(32,32), kernel_size=2, dilations=(1,2), dropout=.1, output_channels=8):
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels,hidden_channels[0],1)
        blocks = []
        width = hidden_channels[0]
        for out_channels, dilation in zip(hidden_channels,dilations,strict=True):
            blocks.append(TemporalBlock(width,out_channels,kernel_size,dilation,dropout))
            width = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(width,output_channels)

    @property
    def receptive_field(self):
        # Sum the actual sequential causal layers; projections have kernel 1.
        return 1 + sum((layer.conv.kernel_size[0]-1)*layer.conv.dilation[0] for layer in self.modules() if isinstance(layer,CausalConv1d))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        hidden = self.blocks(self.input_projection(x.transpose(1,2)))
        logits = self.head(hidden.transpose(1,2))
        if not torch.isfinite(logits).all():
            raise ValueError('non-finite logits')
        return logits


def masked_bce_with_logits(logits, targets, padding_mask):
    denominator = padding_mask.sum()*logits.shape[-1]
    if denominator <= 0:
        raise ValueError('no real label positions')
    losses = F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    loss = (losses*padding_mask[...,None]).sum()/denominator
    if not torch.isfinite(loss):
        raise ValueError('non-finite loss')
    return loss


def model_state_hash(model):
    h = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        meta = json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode('utf-8')
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        for item in (meta,raw):
            h.update(len(item).to_bytes(8,'big'))
            h.update(item)
    return 'sha256:' + h.hexdigest()
