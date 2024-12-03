from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions

input_shape =( 640, 640)

def load_model():
    #model=tf.keras.Applications.MobileNetV2(input_shape)
    model=tf.saved_model.load('exported_model')
    infer = model.signatures["serving_default"]
    input_tensor = tf.random.uniform([1, 640, 640, 3], dtype=tf.float32)
    detections=infer(input_tensor=input_tensor)

_model=load_model()

def preprocess(image: Image.Image):
    image=image.resize(input_shape)
    image=np.array(image).astype('float32')/255.0
    image=np.expand_dims(image, 0)
    return image

#def predict(image: np.ndarray):
    #predictions=_model.predict(image)
    #predictions=imagenet_utils.decode_predictions(predictions)[0][0][1]
    #return predictions

def predict(image: np.ndarray):
    predictions = _model(image)
    return predictions

def read_image(image_encoded):
    pil_image=Image.open(BytesIO(image_encoded))
    return pil_image
