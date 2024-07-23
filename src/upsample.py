import os
import torch
import numpy as np
import torch.nn as nn
from utils.util import draw

class Upsample(nn.Module):
    def __init__(
            self, in_chans, out_chans, kernel_size=2, stride=2, 
            padding=0, tissue_name = "UNDEFINED", device='cuda'
        ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            device=device
        )

        self.upsample.weight = nn.Parameter(torch.rand(
            [out_chans, in_chans, kernel_size, kernel_size]
        ).to(self.upsample.weight.device))
        self.upsample.bias = nn.Parameter(
            torch.rand([out_chans]).to(self.upsample.bias.device)
        )

        self.intermediate_tensor = None
        self.groud_truth = None
        self.tissue_name = tissue_name

    def forward(self, input):
        self.intermediate_tensor = self.upsample(input)
        return self.intermediate_tensor

    def save_plots(self, plots_dir):
        save_path = os.path.join(plots_dir, f"upsampled_{self.tissue_name}.png")
        if isinstance(self.groud_truth, type(None)):
            raise RuntimeError("Ground is not assigned yet!")
        elif not isinstance(self.groud_truth, np.ndarray):
            raise RuntimeError("Ground truth is not a NumPy array!")
        
        clone = self.intermediate_tensor.clone().detach().cpu().squeeze().unsqueeze(-1).numpy()
        draw(self.groud_truth, clone, gt_name=f"{self.tissue_name}_target",
                pred_name=f"{self.tissue_name} Predicted", save_at=save_path)

