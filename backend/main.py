from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def home():
    return {"message": "MSLI Backend is running"}