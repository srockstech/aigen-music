#!/usr/bin/env python3

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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
    title="MusicGen API",
    description="AI Music Generation API powered by Meta's MusicGen",
    version="1.0.0"
)

# Create output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Initialize model
try:
    logger.info("Loading MusicGen model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "facebook/musicgen-small"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Mount static files
app.mount("/files", StaticFiles(directory=str(OUTPUTS_DIR)), name="files")

class GenerationParams(BaseModel):
    """Parameters for music generation"""
    text: str = Field(..., description="Text prompt describing the desired music")
    duration: Optional[int] = Field(
        default=10,
        ge=1,
        le=30,
        description="Duration of generated audio in seconds"
    )
    guidance_scale: Optional[float] = Field(
        default=3.0,
        gt=0.0,
        le=10.0,
        description="Classifier-free guidance scale (higher = more adherence to text)"
    )
    temperature: Optional[float] = Field(
        default=1.0,
        gt=0.0,
        le=2.0,
        description="Sampling temperature (higher = more random)"
    )

class GenerationResponse(BaseModel):
    """Response from music generation endpoint"""
    file_url: str
    duration: int
    timestamp: str
    prompt: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "musicgen-small",
        "device": str(device)
    }

@app.post("/generate/", response_model=GenerationResponse)
async def generate(params: GenerationParams):
    """Generate music from text prompt"""
    try:
        logger.info(f"Generating audio for prompt: {params.text}")
        
        # Process the text input
        inputs = processor(
            text=[params.text],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate audio
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=params.guidance_scale,
                max_new_tokens=256 * params.duration,  # Adjust based on duration
                temperature=params.temperature
            )
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUTS_DIR / filename
        
        # Save audio file
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0].cpu().numpy()
        
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