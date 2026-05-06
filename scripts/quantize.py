from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "models/mobilenetv3.onnx",
    "models/mobilenetv3_quant.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized")