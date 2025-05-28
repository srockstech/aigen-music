#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path
import torch
from audiocraft.data.audio import AudioDataset
from audiocraft.models import MusicGen
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

def setup_logging(log_dir: str) -> None:
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training.log"),
            logging.StreamHandler()
        ]
    )

def train(args):
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        logger.info(f"Loading MusicGen model: {args.model}")
        model = MusicGen.get_pretrained(args.model)
        model.to(device)
        
        # Load dataset
        logger.info(f"Loading dataset from: {args.dataset}")
        ds = AudioDataset(
            args.dataset,
            duration=args.duration,
            text_column="caption",
            sample_rate=32000  # MusicGen's required sample rate
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Setup optimizer
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Training loop
        logger.info("Starting training...")
        best_loss = float('inf')
        
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            progress_bar = tqdm(dl, desc=f"Epoch {epoch + 1}/{args.epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                # Move batch to device
                audio = batch['audio'].to(device)
                text = batch['text']
                
                # Forward pass
                loss = model.compute_loss(audio, text)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                
                # Update progress
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Save checkpoint periodically
                if (batch_idx + 1) % args.save_every == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pt")
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(dl)
            logger.info(f"Epoch {epoch + 1}/{args.epochs} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MusicGen model")
    parser.add_argument("--model", default="small", choices=["small", "medium", "large", "melody"],
                      help="MusicGen model size to use")
    parser.add_argument("--dataset", required=True,
                      help="Path to dataset directory")
    parser.add_argument("--duration", type=int, default=10,
                      help="Duration of audio clips in seconds")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                      help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--output_dir", default="outputs",
                      help="Directory to save checkpoints and logs")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of data loading workers")
    parser.add_argument("--save_every", type=int, default=100,
                      help="Save checkpoint every N batches")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm for clipping")
    
    args = parser.parse_args()
    train(args) 