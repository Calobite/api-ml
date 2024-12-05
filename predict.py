import tflite_runtime.interpreter as tflite

from helpers import read_labels

MODEL_PATH = 'models/detect.tflite'
LABEL_PATH = 'models/labels.txt'
MIN_TRESHOLD = 0.5
INPUT_MEAN = 127.5
INPUT_STD = 127.5

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
labels = read_labels(LABEL_PATH)
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]