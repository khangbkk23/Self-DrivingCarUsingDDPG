# ddpg/models.py
import torch
import torch.nn as nn
import numpy as np

class CNNEncoder(nn.Module):
    def __init__(self, config):
        super(CNNEncoder, self).__init__()
        
        in_channels = config['cnn_model']['in_channels']
        out_channels = config['cnn_model']['out_channels']
        kernel_sizes = config['cnn_model']['kernel_sizes']
        stride = config['cnn_model']['stride']
        padding = config['cnn_model']['paddings']
        
        modules = []
        current_in_channels = in_channels

        for i, out_c in enumerate(out_channels):
            modules.append(nn.Sequential(
                nn.Conv2d(
                    current_in_channels, 
                    out_c, 
                    kernel_size=kernel_sizes[i] if isinstance(kernel_sizes, list) else kernel_sizes, 
                    stride=stride, 
                    padding=padding
                ),
                nn.ReLU(),
                nn.BatchNorm2d(out_c)
            ))
            current_in_channels = out_c
            
        self.cnn = nn.Sequential(*modules)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 96, 96) # 96x96
            dummy_output = self.cnn(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).shape[1]
            
        print(f"CNN Output flatten dim: {self.flatten_dim}")

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1) # Flatten: (Batch, C, H, W) -> (Batch, Vector)
        return x