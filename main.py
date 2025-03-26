from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import replicate
import requests
import os
from dotenv import load_dotenv
from io import BytesIO
import asyncio
from starlette.background import BackgroundTask

load_dotenv()
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
app = FastAPI(title="Interior design api")

# Global response cache for storing generated images
response_cache = {}

def cleanup_cache(image_id):
    """Remove completed image from cache after it's been served"""
    if image_id in response_cache:
        del response_cache[image_id]

async def process_image(image_id: str, input_file: bytes, steps: int = 40, prompt: str = "") -> None:
    """Process image in background and store result in cache"""
    try:
        print(f"Processing image {image_id}...")
        image_file = BytesIO(input_file)
        input_data = {
            "image": image_file,
            "num_inference_steps": steps,
            "prompt": prompt
        }
        
        # Call the Replicate API
        output = replicate.run(
            "adirik/interior-design:76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38",
            input=input_data
        )
        
        # Handle different possible output formats from Replicate
        if isinstance(output, list) and len(output) > 0:
            image_url = output[0]
        elif isinstance(output, dict) and 'image' in output:
            image_url = output['image']
        else:
            image_url = output
            
        print(f"Generated Image URL: {image_url}")
        
        # Don't validate the URL format, just try to fetch it
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Will raise an exception if status code is not 200
            response_cache[image_id] = response.content
            print(f"Successfully processed image {image_id}")
        except requests.RequestException as req_err:
            raise ValueError(f"Failed to fetch image from URL: {str(req_err)}")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        # Store the error in the cache
        response_cache[image_id] = f"ERROR: {str(e)}"

@app.post("/generate-image/")
async def generate_image(file: UploadFile = File(...), prompt: str = ""):
    try:
        # Generate a unique ID for this request
        import uuid
        image_id = str(uuid.uuid4())
        
        # Read the image
        image_bytes = await file.read()
        
        # Start image processing in the background
        asyncio.create_task(process_image(image_id, image_bytes, prompt=prompt))
        
        # Wait a short time for the process to begin
        await asyncio.sleep(1)
        
        # Try to get the result (it might not be ready yet)
        max_retries = 30  # Maximum 60 seconds wait (30 retries * 2 seconds)
        for i in range(max_retries):
            if image_id in response_cache:
                result = response_cache[image_id]
                
                # Check if it's an error
                if isinstance(result, str) and result.startswith("ERROR:"):
                    raise HTTPException(status_code=500, detail=result[6:])
                
                # Return the image with a cleanup task
                return StreamingResponse(
                    BytesIO(result), 
                    media_type="image/png",
                    background=BackgroundTask(cleanup_cache, image_id)
                )
            
            # Wait a bit before checking again
            await asyncio.sleep(2)
            print(f"Waiting for image {image_id}, retry {i+1}/{max_retries}")
        
        # If we've waited too long, return a 202 Accepted status
        return StreamingResponse(
            BytesIO(b"Processing image. Please retry in a few seconds."),
            media_type="text/plain",
            status_code=202
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))