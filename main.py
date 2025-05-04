from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import base64
import io
from PIL import Image
import requests
import os

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your extension origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR setup
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Mistral API info
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Base64 image schema
class ImageData(BaseModel):
    image: str  # base64 encoded image

# Extract text using OCR
def extract_text_from_image(image_bytes: bytes) -> str:
    with open("temp.png", "wb") as f:
        f.write(image_bytes)
    result = ocr.ocr("temp.png", cls=True)
    lines = [line[1][0] for line in result[0]]
    return " ".join(lines)

# Query Mistral
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

# Solve endpoint
@app.post("/solve")
async def solve_math_image(data: ImageData):
    try:
        # Decode base64 image
        base64_data = data.image.split(",")[-1]  # Strip 'data:image/...;base64,' if present
        image_bytes = base64.b64decode(base64_data)

        # OCR
        question_text = extract_text_from_image(image_bytes)
        if not question_text:
            return JSONResponse(status_code=400, content={"error": "No text found in image."})

        # Ask Mistral
        answer = ask_mistral(question_text)
        return {
            "question": question_text,
            "answer": answer
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Math Solver API is running"}
