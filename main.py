from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import base64, io
from PIL import Image
import requests
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 In production, replace with your actual extension ID
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize PaddleOCR (English, enable angle detection)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Get your Mistral API key from environment variable
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def extract_text_from_image(image_bytes: bytes) -> str:
    with open("temp.png", "wb") as f:
        f.write(image_bytes)
    result = ocr.ocr("temp.png", cls=True)
    lines = [line[1][0] for line in result[0]]
    return " ".join(lines)

def ask_mistral(question: str) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-medium",
        "messages": [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": f"Solve this step-by-step: {question}"}
        ]
    }
    response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
    data = response.json()
    return data['choices'][0]['message']['content']

@app.post("/solve")
async def solve_math_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    question_text = extract_text_from_image(image_bytes)
    if not question_text:
        return JSONResponse(status_code=400, content={"error": "No text found in image."})

    answer = ask_mistral(question_text)
    return {
        "question": question_text,
        "answer": answer
    }
