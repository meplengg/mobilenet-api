import onnxruntime as ort
import numpy as np

class Model:

    def __init__(self):

        self.session = ort.InferenceSession(
            "models/mobilenetv3_quant.onnx",
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        # โหลด labels
        with open("imagenet_classes.txt") as f:
            self.labels = [line.strip() for line in f]

    def predict(self, x):

        outputs = self.session.run(
            None,
            {
                self.input_name: x.astype(np.float32)
            }
        )

        pred = np.argmax(outputs[0], axis=1)[0]

        label = self.labels[pred]

        return {
            "class_id": int(pred),
            "label": label
        }