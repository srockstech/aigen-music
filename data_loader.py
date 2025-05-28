#!/usr/bin/env python3

import os
import torchaudio
from torch.utils.data import Dataset
from typing import Tuple, List
import torch

class FMAFloat32(Dataset):
    """Dataset loader for Free Music Archive (FMA) dataset.
    
    Args:
        root_dir (str): Root directory containing the FMA dataset
        split (str): Dataset split to use ('small', 'medium', or 'large')
        target_sr (int, optional): Target sample rate for audio. Defaults to 32000.
    """
    
    def __init__(self, root_dir: str, split: str = "small", target_sr: int = 32000):
        self.root_dir = root_dir
        self.target_sr = target_sr
        
        # Validate split
        if split not in ["small", "medium", "large"]:
            raise ValueError(f"Invalid split '{split}'. Must be one of: small, medium, large")
        
        # Load manifest file
        manifest_path = os.path.join(root_dir, f"fma_{split}_list.txt")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            
        with open(manifest_path) as f:
            self.files = [os.path.join(root_dir, p.strip()) for p in f.readlines()]
            
        # Validate that we have files
        if not self.files:
            raise RuntimeError(f"No audio files found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and process an audio file.
        
        Args:
            idx (int): Index of the audio file to load
            
        Returns:
            Tuple[torch.Tensor, int]: Audio waveform and sample rate
            
        Raises:
            RuntimeError: If audio file cannot be loaded
        """
        try:
            wav, sr = torchaudio.load(self.files[idx])
            
            # Resample if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                wav = resampler(wav)
                sr = self.target_sr
                
            return wav, sr
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {self.files[idx]}: {str(e)}")

    def get_audio_paths(self) -> List[str]:
        """Get list of all audio file paths in the dataset.
        
        Returns:
            List[str]: List of audio file paths
        """
        return self.files.copy()

# Example usage:
# loader = FMAFloat32("/data/fma", split="small", target_sr=32000) 