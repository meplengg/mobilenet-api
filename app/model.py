import timm
import torch
import numpy as np

class Model:
    def __init__(self):
        self.model = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=False
        )
        self.model.eval()

    def predict(self, x):
        x = torch.tensor(x)

        with torch.no_grad():
            outputs = self.model(x)

        pred = outputs.argmax(dim=1).item()

        return int(pred)