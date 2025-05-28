#!/usr/bin/env python3

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import asyncio
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="S.Rocks.Music AI Generation API",
    description="""
    An AI-powered music generation API using Meta's MusicGen model.
    
    This API allows you to:
    * Generate music from text descriptions
    * Control generation parameters like duration and style
    * Download generated audio files
    * List previously generated music
    
    For best results:
    * Be specific in your text descriptions
    * Include instruments, mood, and genre
    * Keep descriptions under 100 words
    * Start with shorter durations (10-30 seconds)
    """,
    version="1.0.0",
    contact={
        "name": "Sarvagya Sharma",
        "url": "https://github.com/srockstech/aigen-music"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Model configuration
MAX_DURATION = 30  # Maximum duration in seconds
SAMPLE_RATE = 32000  # Audio sample rate
TOKENS_PER_SECOND = 50  # Approximate number of tokens per second
MAX_NEW_TOKENS = 1024  # Maximum number of tokens for generation

# Global variables for model state
model = None
processor = None
model_ready = False
model_error = None

def load_model():
    """Load the model in a separate thread"""
    global model, processor, model_ready, model_error
    try:
        logger.info("Starting model loading process...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model_id = "facebook/musicgen-small"
        processor = AutoProcessor.from_pretrained(model_id)
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )
        model.to(device)
        
        model_ready = True
        logger.info("Model loaded successfully!")
    except Exception as e:
        model_error = str(e)
        logger.error(f"Failed to load model: {str(e)}")
        model_ready = False

# Start model loading in background
threading.Thread(target=load_model, daemon=True).start()

# Mount static files
app.mount("/files", StaticFiles(directory=str(OUTPUTS_DIR)), name="files")

class GenerationParams(BaseModel):
    """Parameters for music generation"""
    text: str = Field(
        ..., 
        description="Text prompt describing the desired music",
        example="A lofi hip hop beat with smooth jazz piano and rain sounds",
        min_length=3,
        max_length=500
    )
    duration: Optional[int] = Field(
        default=10,
        ge=1,
        le=MAX_DURATION,
        description=f"Duration of generated audio in seconds (max {MAX_DURATION})",
        example=15
    )
    guidance_scale: Optional[float] = Field(
        default=3.0,
        gt=0.0,
        le=10.0,
        description="Classifier-free guidance scale (higher = more adherence to text)",
        example=3.0
    )
    temperature: Optional[float] = Field(
        default=1.0,
        gt=0.0,
        le=2.0,
        description="Sampling temperature (higher = more random)",
        example=1.0
    )

class GenerationResponse(BaseModel):
    """Response from music generation endpoint"""
    file_url: str = Field(..., description="URL to download the generated audio file")
    duration: int = Field(..., description="Duration of the generated audio in seconds")
    timestamp: str = Field(..., description="Generation timestamp")
    prompt: str = Field(..., description="Original text prompt used for generation")

def calculate_max_new_tokens(duration: int) -> int:
    """Calculate the number of tokens needed for the desired duration."""
    tokens = min(int(duration * TOKENS_PER_SECOND), MAX_NEW_TOKENS)
    logger.info(f"Calculated tokens for {duration}s duration: {tokens}")
    return tokens

@app.get("/", 
    response_model=dict,
    summary="Health Check",
    description="Check if the API is running and get model loading status"
)
async def root():
    """Health check endpoint that responds immediately"""
    status = {
        "status": "starting" if not model_ready and not model_error else "healthy" if model_ready else "error",
        "model_status": "loading" if not model_ready and not model_error else "ready" if model_ready else "failed",
        "error": model_error if model_error else None,
        "version": "1.0.0"
    }
    
    # Return 200 even if model is loading to pass Railway health checks
    return status

@app.get("/status",
    response_model=dict,
    summary="Detailed Status",
    description="Get detailed status of the API and model"
)
async def status():
    """Detailed status endpoint"""
    return {
        "status": "healthy" if model_ready else "starting",
        "model": "musicgen-small",
        "model_status": "ready" if model_ready else "loading",
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "error": model_error if model_error else None,
        "version": "1.0.0"
    }

@app.post("/generate/", 
    response_model=GenerationResponse,
    summary="Generate Music",
    description="""
    Generate music from a text description.
    
    Example prompts:
    * "A lofi hip hop beat with smooth jazz piano and rain sounds"
    * "Epic orchestral music with dramatic strings and powerful drums"
    * "Ambient electronic music with synth pads and gentle beats"
    * "Traditional Indian classical music with sitar and tabla"
    """
)
async def generate(params: GenerationParams):
    """Generate music from text prompt"""
    if not model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please try again in a few moments."
        )
    
    try:
        logger.info(f"Generating audio for prompt: {params.text}")
        
        # Calculate tokens based on duration
        max_new_tokens = calculate_max_new_tokens(params.duration)
        logger.info(f"Using max_new_tokens: {max_new_tokens}")
        
        # Process the text input
        inputs = processor(
            text=[params.text],
            padding=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate audio with safety checks
        try:
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=params.guidance_scale,
                    max_new_tokens=max_new_tokens,
                    temperature=params.temperature
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise HTTPException(
                    status_code=503,
                    detail="Server is out of memory. Try a shorter duration or wait a moment."
                )
            raise
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUTS_DIR / filename
        
        # Save audio file
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        
        import scipy.io.wavfile
        scipy.io.wavfile.write(
            str(output_path),
            rate=sampling_rate,
            data=audio_data
        )
        
        logger.info(f"Audio saved to {output_path}")
        
        # Clean up old files if more than 100 files
        cleanup_old_files()
        
        return GenerationResponse(
            file_url=f"/files/{filename}",
            duration=params.duration,
            timestamp=timestamp,
            prompt=params.text
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate audio: {str(e)}"
        )

def cleanup_old_files(max_files: int = 100):
    """Clean up old generated files if too many exist"""
    try:
        files = list(OUTPUTS_DIR.glob("*.wav"))
        if len(files) > max_files:
            # Sort by creation time and remove oldest
            files.sort(key=lambda x: x.stat().st_ctime)
            for file in files[:-max_files]:
                file.unlink()
                logger.info(f"Cleaned up old file: {file}")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 