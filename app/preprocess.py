from PIL import Image
import numpy as np

def preprocess(image: Image.Image):
    image = image.resize((224,224))
    image = np.array(image).astype("float32") / 255.0

    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std = np.array([0.229,0.224,0.225], dtype=np.float32)

    image = (image - mean) / std

    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image,axis=0)

    return image.astype("float32")   # ✅ สำคัญ