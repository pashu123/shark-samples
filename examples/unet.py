import torch
import numpy as np
import os
import sys
from shark_runner import shark_inference


class UnetModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )
        self.train(False)

    def forward(self, input):
        return self.model(input)


input = torch.randn(1, 3, 224, 224)

results = shark_inference(
    UnetModule(),
    input,
    device="cpu",
    dynamic=False,
    jit_trace=False,
)
