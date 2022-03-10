import torch
import numpy as np
import os
import sys
from shark_runner import shark_inference


class ResNest50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "zhanghang1989/ResNeSt", "resnest50", pretrained=True
        )
        self.train(False)

    def forward(self, input):
        return self.model.forward(input)


input = torch.randn(1, 3, 224, 224)

results = shark_inference(
    ResNest50(),
    input,
    device="cpu",
    dynamic=False,
    jit_trace=True,
)
