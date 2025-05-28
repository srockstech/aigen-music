#!/usr/bin/env python3

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import time

def generate_music(prompt, duration=10, output_path=None):
    """
    Generate music using MusicGen model
    """
    print("Loading MusicGen model...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    
    # Set the duration (approximately 50 tokens per second)
    model.generation_config.max_new_tokens = int(duration * 50)
    
    print(f"Generating music for prompt: {prompt}")
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    # Generate audio
    audio_values = model.generate(**inputs, do_sample=True)
    audio_values = audio_values.cpu().numpy().squeeze()
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"generated_music_{timestamp}.wav"
    
    # Save audio file
    sample_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_values)
    print(f"Music saved to: {output_path}")
    return output_path

def main():
    # Example prompts for Indian music
    prompts = [
        # Classical
        "Indian classical raga in morning, peaceful with tanpura drone and slow tabla",
        
        # Fusion
        "Modern Indian fusion with electronic beats, sitar melody and ambient synths",
        
        # Folk
        "Upbeat Indian folk music with dholak rhythm and harmonium melody",
        
        # Contemporary
        "Bollywood style dance music with tabla and electronic drums"
    ]
    
    # Generate music for each prompt
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating sample {i+1}:")
        output_path = f"indian_music_sample_{i+1}.wav"
        generate_music(prompt, duration=15, output_path=output_path)
        print("-" * 50)

if __name__ == "__main__":
    main() 