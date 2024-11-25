from fastapi import FastAPI
import uvicorn
from fastapi import File, UploadFile
from prediction import read_image
from prediction import preprocess
from prediction import predict

app = FastAPI()

@app.post('predict')
async def predict_image(file: UploadFile = File(...)):
    image = read_image(await file)
    image = preprocess(image)

    predictions = predict(image)
    print(predictions)
    return predictions

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')