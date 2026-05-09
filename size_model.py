import torch
import timm

model = timm.create_model(
    "mobilenetv3_large_100.ra_in1k",
    pretrained=True
)

torch.save(
    model.state_dict(),
    "models/mobilenetv3_original.pth"
)

print("Saved")