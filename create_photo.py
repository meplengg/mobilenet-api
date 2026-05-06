from PIL import Image
import numpy as np

img = Image.fromarray(
    (np.random.rand(224,224,3)*255).astype('uint8')
)
img.save("test.jpg")