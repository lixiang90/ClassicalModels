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
from vae.discriminator import NLayerDiscriminator, weights_init
from vae.losses import LPIPS, hinge_d_loss, GradientLoss

# Hyperparameters
BATCH_SIZE = 4 # Small batch size due to memory constraints
LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 5e-4
NUM_STEPS = 200000 # Extended steps
SAVE_EVERY = 10000
SAMPLE_EVERY = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "E:/pythonfiles/megaDiT/imagenet-dataset/data"

# Loss Weights
REC_WEIGHT = 1.0
KL_WEIGHT = 1e-6
PERCEPTUAL_WEIGHT = 1.0
DISC_WEIGHT = 0.5
GRADIENT_WEIGHT = 0.0 # Weight for gradient/text loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (optional)")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print("Training: VAE + GAN + Perceptual Loss")
    
    # 1. Initialize Models
    # VAE
    vae = VAE(in_channels=3, z_channels=4, ch=64, ch_mult=(1, 2, 4, 4)).to(DEVICE)
    # Discriminator
    discriminator = NLayerDiscriminator(input_nc=3, n_layers=3).to(DEVICE)
    discriminator.apply(weights_init) # Initialize weights
    # Perceptual Loss
    perceptual_loss = LPIPS().to(DEVICE).eval()
    # Gradient Loss (for text/edges)
    gradient_loss = GradientLoss(channels=3, device=DEVICE).to(DEVICE)
    
    # Optimizers
    opt_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    opt_disc = optim.Adam(discriminator.parameters(), lr=DISC_LEARNING_RATE, betas=(0.5, 0.9))
    
    # Load checkpoint if provided
    start_step = 0
    
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            vae.load_state_dict(checkpoint["model"])
            
            # Check if this is a checkpoint with discriminator (from previous run of this script)
            if "discriminator" in checkpoint:
                print("Detected checkpoint with discriminator. Resuming training...")
                discriminator.load_state_dict(checkpoint["discriminator"])
                if "optimizer_vae" in checkpoint:
                    opt_vae.load_state_dict(checkpoint["optimizer_vae"])
                if "optimizer_disc" in checkpoint:
                    opt_disc.load_state_dict(checkpoint["optimizer_disc"])
                if "step" in checkpoint:
                    start_step = checkpoint["step"]
                    print(f"Resumed from step {start_step}")
            else:
                print("Detected VAE-only checkpoint. Starting GAN training from step 0 with loaded VAE weights.")
                # Only VAE weights loaded, train from scratch for GAN components
                start_step = 0
        else:
            raise ValueError("Invalid checkpoint format. Expected dict with 'model' key.")
    else:
        print("No checkpoint provided. Training from scratch.")

    # 2. Setup Data Loading
    train_files = glob.glob(os.path.join(DATA_DIR, "train-*.parquet"))
    if not train_files:
        raise ValueError(f"No training files found in {DATA_DIR}")
        
    print(f"Found {len(train_files)} training files.")
    
    # Streaming load
    dataset = load_dataset("parquet", data_files=train_files, split="train", streaming=True)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])
    
    def preprocess(examples):
        pixel_values = []
        for image in examples["image"]:
            if image.mode != "RGB":
                image = image.convert("RGB")
            pixel_values.append(transform(image))
        return {"pixel_values": pixel_values}

    dataset = dataset.map(preprocess, batched=True, batch_size=BATCH_SIZE)
    
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        return pixel_values
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Skip batches if needed
    if start_step > 0:
        print(f"Skipping {start_step * BATCH_SIZE} samples...")
        try:
            dataset = dataset.skip(start_step * BATCH_SIZE)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
            print("Successfully skipped data.")
        except Exception as e:
            print(f"Skip failed: {e}. Consuming iterator manually...")
            temp_iter = iter(dataloader)
            for _ in tqdm(range(start_step)):
                next(temp_iter)
            dataloader = temp_iter

    # 3. Training Loop
    vae.train()
    discriminator.train()
    
    progress_bar = tqdm(initial=start_step, total=NUM_STEPS)
    data_iter = iter(dataloader)
    step = start_step
    
    while step < NUM_STEPS:
        try:
            images = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images = next(data_iter)
            
        images = images.to(DEVICE)
        
        # --- Generator (VAE) Step ---
        # Disable Discriminator grads
        for p in discriminator.parameters():
            p.requires_grad = False
            
        opt_vae.zero_grad()
        
        # Forward pass
        reconstructions, posterior = vae(images)
        
        # 1. Reconstruction Loss (L1)
        rec_loss = torch.abs(images - reconstructions).sum() # Sum reduction
        n_pixels = images.shape[0] * images.shape[1] * images.shape[2] * images.shape[3]
        rec_loss_mean = rec_loss / n_pixels # For logging
        
        # 2. Perceptual Loss (LPIPS)
        # LPIPS expects input in [-1, 1] (our data is already [-1, 1])
        p_loss = perceptual_loss(images, reconstructions).sum()
        
        # 3. KL Divergence
        kl_loss = posterior.kl().sum()
        
        # 4. GAN Loss (Generator part)
        # We want discriminator to classify reconstructions as Real (1)
        logits_fake = discriminator(reconstructions)
        g_loss = -torch.mean(logits_fake) # Hinge loss generator part
        
        # 5. Gradient Loss (Text/Edges)
        grad_loss = gradient_loss(reconstructions, images)
        
        # Total Generator Loss
        # Weights need careful tuning. Here we use standard latent diffusion weights
        # Usually rec_weight is 1.0, perceptual_weight 1.0
        # However, since we use SUM reduction for Rec/KL, we need to balance them.
        # Let's use MEAN for weighted sum logic to be easier, but user code used SUM.
        # To adapt: rec_loss is huge (e.g. 1e4). p_loss is usually small (0.x * batch).
        # We should probably normalize rec_loss or scale p_loss up.
        # Strategy: Use the same logic as previous stage but add P_loss and G_loss
        
        # Re-scaling to match previous stage magnitude
        # Previous stage: loss = rec_loss + kl_weight * kl_loss
        # Now: loss = rec_loss + kl_weight * kl_loss + p_weight * p_loss + g_weight * g_loss
        # But p_loss and g_loss are mean-based usually.
        # Let's scale p_loss and g_loss by number of pixels to match rec_loss magnitude (sum)
        # Or just use weights.
        
        # Better approach for Stage 2:
        # Use a balanced formulation.
        # rec_loss is sum. p_loss is sum. kl_loss is sum.
        # g_loss is mean. -> Convert to sum
        g_loss_sum = g_loss * images.shape[0] 
        grad_loss_sum = grad_loss * images.shape[0] * images.shape[1] * images.shape[2] * images.shape[3] # L1 is mean by default in GradientLoss? No, check impl.
        # GradientLoss uses F.l1_loss which defaults to mean. So we scale it up.
        
        # We need to be careful. Let's use standard weights but applied to mean losses?
        # No, let's stick to what we have but add new terms.
        
        # rec_loss ~ 20000 (for batch 4, 256x256)
        # p_loss ~ 1.0 * 4 = 4.0
        # So we need to scale p_loss by factor ~5000? 
        # Actually, in standard VQGAN, rec_loss is L1.
        # Let's just trust the weights: 
        # If we want P_loss to matter, it needs to be comparable.
        # Let's treat everything as weighted sum.
        
        total_gen_loss = (REC_WEIGHT * rec_loss) + \
                         (KL_WEIGHT * kl_loss) + \
                         (PERCEPTUAL_WEIGHT * p_loss * 1000) + \
                         (DISC_WEIGHT * g_loss_sum * 1000) + \
                         (GRADIENT_WEIGHT * grad_loss_sum * 10)
                         # *1000 is a heuristic to make P/G loss comparable to pixel-sum L1
        
        total_gen_loss.backward()
        opt_vae.step()
        
        # --- Discriminator Step ---
        # Enable Discriminator grads
        for p in discriminator.parameters():
            p.requires_grad = True
            
        opt_disc.zero_grad()
        
        logits_real = discriminator(images.detach())
        logits_fake = discriminator(reconstructions.detach())
        
        # Hinge Loss
        d_loss = hinge_d_loss(logits_real, logits_fake)
        
        d_loss.backward()
        opt_disc.step()
        
        # Logging
        progress_bar.set_description(f"Step {step}")
        progress_bar.set_postfix({
            "rec": f"{rec_loss_mean:.4f}",
            "p_loss": f"{p_loss.item()/BATCH_SIZE:.4f}",
            "g_loss": f"{g_loss.item():.4f}",
            "d_loss": f"{d_loss.item():.4f}",
            "grad": f"{grad_loss.item():.4f}"
        })
        
        step += 1
        progress_bar.update(1)
        
        if step % SAMPLE_EVERY == 0:
            os.makedirs("vae/samples", exist_ok=True)
            n_vis = min(4, images.shape[0])
            orig_vis = images[:n_vis].detach()
            recon_vis = reconstructions[:n_vis].detach()
            comparison = torch.cat([orig_vis, recon_vis], dim=0)
            # Manual denormalize for safety
            comparison = (comparison + 1.0) / 2.0
            comparison = torch.clamp(comparison, 0.0, 1.0)
            save_image(comparison, f"vae/samples/vae_gan_step_{step}.png", nrow=n_vis)

        if step % SAVE_EVERY == 0:
            save_path = f"vae/checkpoints/vae_gan_step_{step}.pt"
            torch.save({
                'step': step,
                'model': vae.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_vae': opt_vae.state_dict(),
                'optimizer_disc': opt_disc.state_dict()
            }, save_path)

    print("Training finished.")
    torch.save({
        'step': step,
        'model': vae.state_dict(),
        'discriminator': discriminator.state_dict()
    }, "vae/checkpoints/vae_gan_final.pt")

if __name__ == "__main__":
    main()
