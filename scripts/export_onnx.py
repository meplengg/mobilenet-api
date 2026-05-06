import torch
import timm

model = timm.create_model(
    'mobilenetv3_large_100.ra_in1k',
    pretrained=True
)

model.eval()

dummy = torch.randn(1,3,224,224)

torch.onnx.export(
    model,
    dummy,
    "models/mobilenetv3.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("Export complete")