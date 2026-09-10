import os
import shutil
import uuid
from fastapi import UploadFile, HTTPException


def validate_video(file: UploadFile):

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Only video files are allowed"
        )


def save_video(file: UploadFile):

    os.makedirs("uploads", exist_ok=True)

    file_extension = os.path.splitext(file.filename)[1]

    unique_filename = f"{uuid.uuid4()}{file_extension}"

    file_path = os.path.join("uploads", unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path