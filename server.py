import io
from PIL import Image

from vgg16_model import CustomVgg16
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/upload")
async def image_upload(file: UploadFile):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    custom_vgg16 = CustomVgg16()
    out = custom_vgg16(img)
    # cnn 결과를 langchain으로 넘긴다.
    return out