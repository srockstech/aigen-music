#!/usr/bin/env python3

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.utils import get_openapi
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

# Create output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Model configuration
MAX_DURATION = 30  # Maximum duration in seconds
SAMPLE_RATE = 32000  # Audio sample rate
TOKENS_PER_SECOND = 50  # Approximate number of tokens per second

# Initialize model
try:
    logger.info("Loading MusicGen model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "facebook/musicgen-small"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    model.to(device)
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

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
    return min(int(duration * TOKENS_PER_SECOND), model.config.max_position_embeddings - 100)

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
    try:
        logger.info(f"Generating audio for prompt: {params.text}")
        
        # Calculate tokens based on duration
        max_new_tokens = calculate_max_new_tokens(params.duration)
        logger.info(f"Calculated max_new_tokens: {max_new_tokens}")
        
        # Process the text input
        inputs = processor(
            text=[params.text],
            padding=True,
            return_tensors="pt",
        )
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
        except IndexError as e:
            if "index out of range" in str(e):
                raise HTTPException(
                    status_code=400,
                    detail=f"Duration too long. Maximum allowed duration is {MAX_DURATION} seconds."
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