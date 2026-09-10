from fastapi import FastAPI, UploadFile, File
from services.model_service import predict
from services.file_service import validate_video, save_video

app = FastAPI()


@app.get("/")
def home():
    return {"message": "MSLI Backend is running"}


@app.get("/health")
def health():
    return {"status": "Backend is healthy"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    validate_video(file)

    file_path = save_video(file)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "saved_path": file_path
    }
    
@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):

    validate_video(file)

    file_path = save_video(file)

    result = predict(file_path)

    return result