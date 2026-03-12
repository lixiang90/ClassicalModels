import argparse
import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm

# Import modules
from dit.model import DiT
from dit.diffusion import Diffusion
from vae.model import VAE

# Constants (Must match training config)
LATENT_SIZE = 32
PATCH_SIZE = 2
IN_CHANNELS = 4
HIDDEN_SIZE = 384
DEPTH = 12
NUM_HEADS = 6
NUM_CLASSES = 1000
LEARN_SIGMA = True
SCALE_FACTOR = 0.18215
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Inference script for DiT (ImageNet)")
    parser.add_argument("--ckpt_path", type=str, default="dit.pt", help="Path to DiT checkpoint")
    parser.add_argument("--vae_path", type=str, default="vae.pt", help="Path to VAE checkpoint")
    parser.add_argument("--output_dir", type=str, default="dit_inference_results", help="Directory to save results")
    parser.add_argument("--classes", type=int, nargs="+", default=[207, 360, 387, 974, 88, 979, 417, 279], help="List of class indices to generate (default: Golden Retriever, Otter, Panda, Geyser, Macaw, Valley, Balloon, Arctic Fox)")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--num_samples_per_class", type=int, default=1, help="Number of samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image_size", type=int, default=256, help="Output image size")
    
    args = parser.parse_args()
    
    # 0. Setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {DEVICE}")

    # 1. Load VAE
    print(f"Loading VAE from {args.vae_path}...")
    vae = VAE(in_channels=3, z_channels=4, ch=64, ch_mult=(1, 2, 4, 4)).to(DEVICE)
    if os.path.exists(args.vae_path):
        checkpoint = torch.load(args.vae_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            vae.load_state_dict(checkpoint["model"])
        else:
            vae.load_state_dict(checkpoint)
    else:
        print(f"Warning: VAE path {args.vae_path} not found. Output will be garbage.")
    
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
        
    # 2. Load DiT
    print(f"Loading DiT from {args.ckpt_path}...")
    dit = DiT(
        input_size=LATENT_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        learn_sigma=LEARN_SIGMA
    ).to(DEVICE)
    
    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=DEVICE)
        # Handle if checkpoint saves 'model' key or just state_dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            dit.load_state_dict(checkpoint["model"])
        else:
            dit.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"DiT checkpoint not found at {args.ckpt_path}")
        
    dit.eval()
    
    # 3. Diffusion
    diffusion = Diffusion(num_diffusion_timesteps=1000, device=DEVICE)
    
    # 4. Generate
    print(f"Generating images for classes: {args.classes}")
    print(f"CFG Scale: {args.cfg_scale}")
    
    # Create labels tensor
    # We want to generate (num_classes * num_samples_per_class) images
    labels = []
    for c in args.classes:
        labels.extend([c] * args.num_samples_per_class)
    
    labels = torch.tensor(labels, device=DEVICE)
    num_total_samples = len(labels)
    
    # Batch generation if too many
    batch_size = 8
    
    for i in range(0, num_total_samples, batch_size):
        batch_labels = labels[i : i + batch_size]
        current_batch_size = len(batch_labels)
        
        print(f"Sampling batch {i // batch_size + 1}/{(num_total_samples + batch_size - 1) // batch_size}...")
        
        with torch.no_grad():
            # Sample latents
            # diffusion.sample expects y to be (B,)
            sampled_latents = diffusion.sample(
                dit, 
                image_size=LATENT_SIZE, 
                batch_size=current_batch_size, 
                y=batch_labels, 
                cfg_scale=args.cfg_scale
            )
            
            # Decode latents
            sampled_latents = sampled_latents / SCALE_FACTOR
            decoded_images = vae.decode(sampled_latents)
            
            # Denormalize
            decoded_images = (decoded_images + 1) / 2
            decoded_images = torch.clamp(decoded_images, 0, 1)
            
            # Save individual images
            for j, img in enumerate(decoded_images):
                global_idx = i + j
                class_idx = batch_labels[j].item()
                save_path = os.path.join(args.output_dir, f"class_{class_idx:04d}_sample_{global_idx:03d}.png")
                save_image(img, save_path)
                print(f"Saved {save_path}")

    print("Done!")

if __name__ == "__main__":
    main()
