import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_dataset
import glob
import os
from tqdm import tqdm
from vae.model import VAE

# Hyperparameters
BATCH_SIZE = 4 # Small batch size due to memory constraints
LEARNING_RATE = 1e-4
NUM_STEPS = 100000 # Number of training steps
SAVE_EVERY = 10000
SAMPLE_EVERY = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "imagenet-dataset/data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    
    # 1. Initialize Model
    model = VAE(in_channels=3, z_channels=4, ch=64, ch_mult=(1, 2, 4, 4)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        
        # Check if checkpoint is full state dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "step" in checkpoint:
                step = checkpoint["step"]
                print(f"Resumed from step {step}")
        else:
            raise ValueError("Invalid checkpoint format. Expected dict with 'model', 'optimizer', and 'step' keys.")

    # 2. Setup Data Loading
    train_files = glob.glob(os.path.join(DATA_DIR, "train-*.parquet"))
    if not train_files:
        raise ValueError(f"No training files found in {DATA_DIR}")
        
    print(f"Found {len(train_files)} training files.")
    
    # Streaming load
    dataset = load_dataset("parquet", data_files={"train": train_files}, split="train", streaming=True)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    def transform_fn(examples):
        # examples["image"] is a list of PIL images
        pixel_values = [transform(image.convert("RGB")) for image in examples["image"]]
        return {"pixel_values": pixel_values}
        
    # Map transform
    dataset = dataset.map(transform_fn, batched=True, batch_size=BATCH_SIZE)
    # Remove unused columns
    dataset = dataset.select_columns(["pixel_values"])
    
    # Create DataLoader
    # Note: datasets.IterableDataset works with DataLoader. 
    # We need a collate_fn to stack tensors because dataset yields dicts with 'pixel_values'
    def collate_fn(batch):
        # items in batch are dicts with 'pixel_values' which are already tensors
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        return pixel_values
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # 3. Training Loop
    model.train()
    # step = 0 # Already initialized above
    progress_bar = tqdm(initial=step, total=NUM_STEPS)
    
    # Skip batches if resuming
    if step > 0:
        print(f"Skipping {step * BATCH_SIZE} samples (approximately {step} batches)...")
        # datasets.IterableDataset.skip() is available but might be slow for large skips
        # Alternatively, we can just consume the iterator
        # For HuggingFace streaming dataset, dataset.skip(n) is efficient if supported
        try:
            # We need to skip samples, not batches, because dataset.skip operates on samples
            dataset = dataset.skip(step * BATCH_SIZE)
            # Re-create dataloader with skipped dataset
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
            print("Successfully skipped data using dataset.skip()")
        except Exception as e:
            print(f"Warning: dataset.skip() failed ({e}). Falling back to manual skipping (this might be slow)...")
            # Fallback: manual consumption
            temp_iter = iter(dataloader)
            for _ in tqdm(range(step), desc="Skipping batches"):
                next(temp_iter)
            dataloader = temp_iter # Note: this might not work as expected with DataLoader re-creation, 
                                   # but for simple IterableDataset it's often better to just use skip()

    # Infinite loop over dataloader (since it's streaming, it might end, but for ImageNet it's huge)
    data_iter = iter(dataloader)
    
    while step < NUM_STEPS:
        try:
            images = next(data_iter)
        except StopIteration:
            # Restart iterator if dataset exhausted (unlikely for ImageNet in 10k steps)
            data_iter = iter(dataloader)
            images = next(data_iter)
            
        images = images.to(DEVICE)
        
        # Forward pass
        # model(x) returns (dec, posterior)
        reconstruction, posterior = model(images)
        
        # Loss calculation
        # Reconstruction loss (MSE or L1)
        rec_loss = torch.abs(images - reconstruction).sum() / images.shape[0] # L1 loss sum over pixels, mean over batch
        # Usually we use sum over pixels for VAE to balance with KL sum
        
        # KL Divergence
        kl_loss = posterior.kl().sum() / images.shape[0]
        
        # Total loss
        # Often we weight KL loss. For VQGAN/LDM, KL weight is small (e.g. 0.000001) but for standard VAE it's 1.0
        # Let's use a small weight for KL if we want high fidelity reconstruction (like in Stable Diffusion autoencoder)
        # But for "standard VAE", usually 1.0 or beta-VAE.
        # Let's stick to standard formulation but maybe print both.
        # If we want good reconstruction, we might need to lower KL weight or use spectral loss etc.
        # For this task "standard VAE", we use standard ELBO: REC + KL
        loss = rec_loss + kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        progress_bar.update(1)
        
        # Calculate per-pixel metrics for logging
        n_pixels = images.shape[1] * images.shape[2] * images.shape[3]
        rec_loss_per_pixel = rec_loss.item() / n_pixels
        
        progress_bar.set_postfix({
            "loss": f"{loss.item():.2e}", 
            "rec_avg": f"{rec_loss_per_pixel:.4f}", 
            "kl": f"{kl_loss.item():.2e}"
        })
        
        if step % SAMPLE_EVERY == 0:
            os.makedirs("vae/samples", exist_ok=True)
            # Take first 4 images
            n_vis = min(4, images.shape[0])
            orig_vis = images[:n_vis].detach()
            recon_vis = reconstruction[:n_vis].detach()
            # Concatenate (Top: Original, Bottom: Reconstruction)
            comparison = torch.cat([orig_vis, recon_vis], dim=0)
            # Manual denormalize
            comparison = (comparison + 1.0) / 2.0
            comparison = torch.clamp(comparison, 0.0, 1.0)
            # Save
            save_image(comparison, f"vae/samples/vae_step_{step}.png", nrow=n_vis)

        if step % SAVE_EVERY == 0:
            save_path = f"vae/checkpoints/vae_step_{step}.pt"
            # Save full state
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)
            # print(f"Saved model to {save_path}")

    print("Training finished.")
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, "vae/checkpoints/vae_final.pt")

if __name__ == "__main__":
    main()
