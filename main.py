from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse,StreamingResponse
import replicate
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import uuid
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
    
    print("Running Stable Diffusion img2img model...")
    output = replicate.run(
        "adirik/interior-design:76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38",
        input=input_data
    )
    
    # Fetch the first generated image from the output URLs
    if not output:
        raise ValueError("No image generated.")
    
    response = requests.get(output[0])
    if response.status_code != 200:
        raise ValueError("Failed to fetch generated image.")
    
    return response.content

@app.post("/generate-image/")
async def generate_image(file: UploadFile = File(...), prompt: str = ""):
    try:
        image_bytes = await file.read()
        generated_image = await process_image(image_bytes, prompt=prompt)
        
        return StreamingResponse(BytesIO(generated_image), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
