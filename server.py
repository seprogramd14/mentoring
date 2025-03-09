# 사진을 올리기 위한 서버 만들기
import io
from PIL import Image
from vgg16_model import CustomVgg16
from chat_model import ChatBot
from fastapi import FastAPI, UploadFile

app = FastAPI()
description = "이 {information}이/가 기숙사에 반입 가능 여부를 반입 가능/불가능으로 나눠서 알려주고 왜 그런지 이유를 한 줄로 설명해줘"

@app.post("/upload")
async def image_upload(file: UploadFile):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    custom_vgg16 = CustomVgg16()
    out = custom_vgg16(img)
    chat_bot = ChatBot(description)
    return chat_bot(out)