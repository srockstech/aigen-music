#!/usr/bin/env python3

import torch
import logging
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Loading MusicGen model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Print PyTorch version and CUDA availability
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Load model and processor with device specified
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            device_map=None,  # Let the model handle device placement
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        model.to(device)
        
        # Set generation parameters
        duration = 30  # reduced to 10 seconds for testing
        logger.info(f"Setting max_new_tokens to {int(duration * 50)}")
        model.generation_config.max_new_tokens = int(duration * 50)  # approximately 50 tokens per second
        
        prompt = "lofi chill beat with tabla and flute"
        logger.info(f"Generating {duration}-second music for prompt: '{prompt}'")
        
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info("Starting audio generation...")
        with torch.no_grad():
            audio_values = model.generate(**inputs)
        
        # Save the audio file
        output_path = "sample.wav"
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        
        logger.info(f"Audio shape: {audio_data.shape}")
        logger.info(f"Sampling rate: {sampling_rate}")
        logger.info(f"Saving audio to {output_path}")
        
        scipy.io.wavfile.write(output_path, sampling_rate, audio_data)
        logger.info(f"Audio successfully saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()