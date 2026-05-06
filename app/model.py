import onnxruntime as ort
import numpy as np

class Model:
    def __init__(self):
        self.session = ort.InferenceSession("models/mobilenetv3.onnx")

    def predict(self, x):
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(
            None,
            {input_name: x}
        )
        return int(np.argmax(outputs[0]))