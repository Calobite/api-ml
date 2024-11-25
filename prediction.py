from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions

input_shape = (1, 640, 640, 3)

def load_model():
    #model=tf.keras.Applications.MobileNetV2(input_shape)
    model=tf.saved_model.load('exported_model')
    infer = model.signatures["serving_default"]
    detections=infer(input)

_model=load_model()

def preprocess(image: Image.image):
    image=image.resize(input_shape)
    image=np.expand_dims(image, 0)
    return image

#def predict(image: np.ndarray):
    #predictions=_model.predict(image)
    #predictions=imagenet_utils.decode_predictions(predictions)[0][0][1]
    #return predictions

def read_image(image_encoded):
    pil_image=Image.open(BytesIO(image_encoded))
    return image_encoded