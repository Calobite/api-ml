from fastapi import FastAPI, File, UploadFile
import uvicorn
from prediction import read_image, preprocess, predict

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML-Model API!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image = read_image(await file.read())
    image = preprocess(image)

    predictions = predict(image)
    print(predictions)
    return {"predictions : ":predictions}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
