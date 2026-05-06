import time
import torch
import timm

model = timm.create_model(
    'mobilenetv3_large_100.ra_in1k',
    pretrained=True
)

model.eval()

x = torch.randn(1,3,224,224)

start = time.time()

with torch.no_grad():
    model(x)

latency = time.time() - start

print("Latency:", latency)