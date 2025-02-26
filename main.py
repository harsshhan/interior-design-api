from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import replicate
import requests
import os
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

app = FastAPI()

async def process_image(input_file: bytes, steps: int = 40, prompt: str = "") -> bytes:
    print("Processing image...")
    image_file = BytesIO(input_file)

    input_data = {
        "image": image_file,
        "num_inference_steps": steps,
        "prompt": prompt
    }
    
    # Call the Replicate API
    image_url = replicate.run(
        "adirik/interior-design:76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38",
        input=input_data
    )
    
    print(f"Generated Image URL: {image_url}")

    if not image_url.startswith("http"):
        raise ValueError(f"Invalid URL received: {image_url}")

    response = requests.get(image_url)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch image from {image_url}")

    return response.content

@app.post("/generate-image/")
async def generate_image(file: UploadFile = File(...), prompt: str = ""):
    try:
        image_bytes = await file.read()
        generated_image = await process_image(image_bytes, prompt=prompt)
        
        return StreamingResponse(BytesIO(generated_image), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))