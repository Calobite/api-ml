from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import io
from PIL import Image
import numpy as np
import sys

from predict import input_height, input_width, interpreter, input_details, output_details, MIN_TRESHOLD, labels, INPUT_MEAN, INPUT_STD


app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML-Model API!"}
    
@app.post("/predict")
async def predict_image( image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        image_width = image.size[0]
        image_height = image.size[1]
        rezied_image = image.resize((input_width, input_height), Image.Resampling.LANCZOS)

        input_data = np.expand_dims(rezied_image, axis=0)
        float_input = (input_details[0]['dtype'] == np.float32)
        if float_input:
          input_data = (np.float32(input_data) - INPUT_MEAN) / INPUT_STD

        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        data = {}
        detections = []

        for i in range(len(scores)):
            detection = {}
            if ((scores[i] > MIN_TRESHOLD) and (scores[i] <= 1.0)):
                detection['ymin'] = int(max(1,(boxes[i][0] * image_height)))
                detection['xmin'] = int(max(1,(boxes[i][1] * image_width)))
                detection['ymax'] = int(min(image_height,(boxes[i][2] * image_height)))
                detection['xmax'] = int(min(image_width,(boxes[i][3] * image_width)))
                detection['score'] = scores[i].item()
                detection['label'] = labels[int(classes[i])]
                
                detections.append(detection)

        data['predictions'] = detections
        data['success'] = True

        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
