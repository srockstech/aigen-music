#!/usr/bin/env python3

# %% [markdown]
# # S.Rocks.Music AI Music Generation Demo
# 
# This notebook demonstrates the capabilities of our AI music generation system using Meta's AudioCraft/MusicGen model.

# %% [markdown]
# ## Setup
# First, let's import the required libraries and set up our model.

# %%
import torch
import logging
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import IPython.display as ipd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %% [markdown]
# ## Model Initialization
# Now we'll load the MusicGen model and set up our device.

# %%
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small",
    device_map=None,  # Let the model handle device placement
    torch_dtype=torch.float32  # Use float32 for CPU
)
model.to(device)

# %% [markdown]
# ## Music Generation Function
# Let's create a function to generate music from text prompts.

# %%
def generate_music(prompt, duration=10, save_path=None):
    """Generate music from a text prompt and return audio data.
    
    Args:
        prompt (str): Text description of the desired music
        duration (int): Duration in seconds (default: 10)
        save_path (str, optional): Path to save the WAV file
        
    Returns:
        tuple: (sampling_rate, audio_data)
    """
    print(f"Generating {duration}-second music for prompt: '{prompt}'")
    
    # Set generation parameters
    model.generation_config.max_new_tokens = int(duration * 50)  # ~50 tokens/second
    
    # Process input
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate audio
    with torch.no_grad():
        audio_values = model.generate(**inputs)
    
    # Process output
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].cpu().numpy()
    
    # Save if path provided
    if save_path:
        scipy.io.wavfile.write(save_path, sampling_rate, audio_data)
        print(f"Audio saved to {save_path}")
    
    return sampling_rate, audio_data

# %% [markdown]
# ## Example 1: Basic Music Generation
# Let's generate a simple lofi beat with tabla and flute.

# %%
prompt = "lofi chill beat with tabla and flute"
sr, audio = generate_music(prompt, duration=10, save_path="demo_sample1.wav")

# Play the audio
ipd.Audio(audio, rate=sr)

# %% [markdown]
# ## Example 2: Different Musical Style
# Now let's try something different - an electronic dance track.

# %%
prompt = "upbeat electronic dance music with synth leads and heavy bass"
sr, audio = generate_music(prompt, duration=10, save_path="demo_sample2.wav")

# Play the audio
ipd.Audio(audio, rate=sr)

# %% [markdown]
# ## Interactive Music Generation
# Use the cell below to generate music with your own prompt!

# %%
from ipywidgets import interact, Text, IntSlider

@interact
def generate_custom_music(prompt=Text(value="jazz piano solo", description="Prompt:"),
                         duration=IntSlider(min=5, max=30, step=5, value=10, description="Duration (s):")):
    sr, audio = generate_music(prompt, duration=duration)
    return ipd.Audio(audio, rate=sr)

# %% [markdown]
# ## Tips for Better Results
# 
# 1. **Be Specific**: Include instruments, mood, genre, and tempo in your prompts
# 2. **Keep it Simple**: Don't make prompts too complex
# 3. **Duration**: Longer durations take more time to generate
# 4. **Experiment**: Try different combinations of instruments and styles
# 
# Example prompts:
# - "ambient pad sounds with gentle piano melodies"
# - "rock guitar riff with drums and bass"
# - "orchestral film score with strings and brass"
# - "traditional indian classical music with sitar" 