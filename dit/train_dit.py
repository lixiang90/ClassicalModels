import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_dataset
import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

# Import DiT modules
from dit.model import DiT
from dit.diffusion import Diffusion
from vae.model import VAE

# Hyperparameters
BATCH_SIZE = 32 # DiT-S/2 is small, can fit more
LEARNING_RATE = 1e-4
NUM_STEPS = 1000000
SAVE_EVERY = 10000
SAMPLE_EVERY = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "imagenet-dataset/data"
VAE_CHECKPOINT = "vae/checkpoints/vae_gan_final.pt"

# DiT Config (DiT-S/2)
PATCH_SIZE = 2
HIDDEN_SIZE = 384
DEPTH = 12
NUM_HEADS = 6

# Latent Scaling Factor (Standard for LDM)
SCALE_FACTOR = 0.18215

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to DiT checkpoint to resume from")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print("Initializing DiT Training...")

    # 1. Load VAE (Frozen)
    print(f"Loading VAE from {VAE_CHECKPOINT}...")
    # Initialize with the same config as training
    vae = VAE(in_channels=3, z_channels=4, ch=64, ch_mult=(1, 2, 4, 4)).to(DEVICE)
    checkpoint = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        vae.load_state_dict(checkpoint["model"])
    else:
        vae.load_state_dict(checkpoint)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE loaded and frozen.")

    # 2. Initialize DiT & Diffusion
    dit = DiT(
        input_size=32, # Latent size (256 / 8)
        patch_size=PATCH_SIZE,
        in_channels=4,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        num_classes=1000,
        learn_sigma=True
    ).to(DEVICE)
    
    diffusion = Diffusion(num_diffusion_timesteps=1000, device=DEVICE)
    
    print(f"DiT Parameters: {sum(p.numel() for p in dit.parameters()):,}")
    
    optimizer = optim.AdamW(dit.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        dit.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")

    # 3. Data Loading (Streaming ImageNet)
    train_files = glob.glob(os.path.join(DATA_DIR, "train-*.parquet"))
    print(f"Found {len(train_files)} training files.")
    
    dataset = load_dataset("parquet", data_files={"train": train_files}, split="train", streaming=True)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    def collate_fn(batch):
        # batch is a list of dicts: {"image": PIL, "label": int}
        images = []
        labels = []
        for item in batch:
            try:
                # Handle potential corrupted images
                if item["image"] is None:
                    continue
                    
                img = item["image"]
                if not isinstance(img, Image.Image):
                    continue
                    
                img = img.convert("RGB")
                images.append(transform(img))
                labels.append(item["label"])
            except Exception as e:
                # print(f"Error processing image: {e}")
                continue
                
        if not images:
            return None
            
        return {
            "pixel_values": torch.stack(images),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Skip if resuming (Approximate)
    if start_step > 0:
        print(f"Note: Resuming streaming dataset. We will just start from the beginning of the stream for simplicity in this script.")
        # Proper skipping in streaming datasets can be slow or complex without a sharded checkpoint system.
        # For this task, we assume it's acceptable to re-train on some data or the user can manually manage shards.
        # Alternatively, we can try to skip:
        # try:
        #     dataset = dataset.skip(start_step * BATCH_SIZE)
        #     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        # except:
        #     pass

    # 4. Training Loop
    dit.train()
    progress_bar = tqdm(initial=start_step, total=NUM_STEPS)
    data_iter = iter(dataloader)
    step = start_step

    os.makedirs("dit/checkpoints", exist_ok=True)
    os.makedirs("dit/samples", exist_ok=True)

    while step < NUM_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                print("Dataset exhausted unexpectedly.")
                break
        except Exception as e:
            print(f"Error fetching batch: {e}")
            continue
        
        if batch is None:
            continue

        images = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # 1. Encode Images to Latents
        with torch.no_grad():
            posterior = vae.encode(images)
            latents = posterior.sample() * SCALE_FACTOR 

        # 2. Diffusion Forward Process
        t = diffusion.sample_timesteps(latents.shape[0])
        noise = torch.randn_like(latents)
        x_t = diffusion.q_sample(latents, t, noise)
        
        # 3. Predict Noise
        # DiT forward: x_t, t, y
        model_output = dit(x_t, t, labels)
        
        # Handle learned sigma output
        if dit.out_channels > dit.in_channels:
            predicted_noise, predicted_var = model_output.chunk(2, dim=1)
        else:
            predicted_noise = model_output

        # Simple MSE Loss on noise
        loss = torch.mean((predicted_noise - noise) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 5. Sampling / Validation
        if step % SAMPLE_EVERY == 0:
            dit.eval()
            print(f"\nSampling at step {step}...")
            # Sample using a fixed set of labels (e.g., first 8 classes or random)
            sample_labels = torch.randint(0, 1000, (8,), device=DEVICE)
            
            # Sample latents
            # Use CFG scale 4.0 for better class coherence
            sampled_latents = diffusion.sample(dit, image_size=32, batch_size=8, y=sample_labels, cfg_scale=4.0)
            
            # Decode latents
            with torch.no_grad():
                # Unscale
                sampled_latents = sampled_latents / SCALE_FACTOR
                decoded_images = vae.decode(sampled_latents)
            
            # Denormalize
            decoded_images = (decoded_images + 1) / 2
            decoded_images = torch.clamp(decoded_images, 0, 1)
            
            save_image(decoded_images, f"dit/samples/step_{step}.png", nrow=4)
            dit.train()

        if step % SAVE_EVERY == 0:
            save_path = f"dit/checkpoints/dit_step_{step}.pt"
            torch.save({
                'step': step,
                'model': dit.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)

    # Final Save
    torch.save({
        'step': step,
        'model': dit.state_dict(),
        'optimizer': optimizer.state_dict()
    }, "dit/checkpoints/dit_final.pt")
    print("Training Finished!")

if __name__ == "__main__":
    main()
