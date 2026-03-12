import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import argparse
import os
import sys

# Ensure current directory is in path so we can import vae
sys.path.append(os.getcwd())

from vae.model import VAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="vae.pt", help="Path to checkpoint")
    parser.add_argument("--image_path", type=str, default="test.png", help="Path to test image")
    parser.add_argument("--output_path", type=str, default="reconstruction.png", help="Path to save reconstruction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # 1. Initialize Model structure
    # Must match training configuration: ch=64, ch_mult=(1, 2, 4, 4)
    vae = VAE(in_channels=3, z_channels=4, ch=64, ch_mult=(1, 2, 4, 4)).to(args.device)
    
    # 2. Load Checkpoint
    if not os.path.exists(args.model_path):
        print(f"Error: Checkpoint not found at {args.model_path}")
        print("Please make sure you have trained the model and have 'vae_gan_final.pt' available.")
        return

    print(f"Loading checkpoint from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            vae.load_state_dict(checkpoint["model"])
        else:
            # Fallback if checkpoint is just the state dict
            vae.load_state_dict(checkpoint)
            
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    vae.eval()

    # 3. Load and Preprocess Image
    if not os.path.exists(args.image_path):
        print(f"Warning: Image not found at {args.image_path}")
        print("Creating a random test image for demonstration...")
        # Create a dummy image
        dummy_img = Image.fromarray(torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8).numpy())
        dummy_img.save(args.image_path)
        print(f"Created {args.image_path}")

    print(f"Loading image from {args.image_path}")
    try:
        img = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    
    # Same transform as training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(args.device) # Add batch dim: [1, 3, 256, 256]

    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        # Encode -> Dist
        posterior = vae.encode(img_tensor)
        
        # Sample latent (z)
        z = posterior.sample()
        # Alternatively use mode for deterministic output: z = posterior.mode()
        
        # Decode -> Reconstruction
        recon = vae.decode(z)

    print(f"Latent shape: {z.shape}") # Should be [1, 4, 32, 32]
    print(f"Reconstruction shape: {recon.shape}")

    # 5. Save Result
    # Denormalize from [-1, 1] to [0, 1]
    recon_vis = (recon + 1.0) / 2.0
    orig_vis = (img_tensor + 1.0) / 2.0
    
    recon_vis = torch.clamp(recon_vis, 0.0, 1.0)
    orig_vis = torch.clamp(orig_vis, 0.0, 1.0)
    
    # Concatenate: Original | Reconstruction
    comparison = torch.cat([orig_vis, recon_vis], dim=3) # Concatenate along width
    
    save_image(comparison, args.output_path)
    print(f"Saved result to {args.output_path} (Left: Original, Right: Reconstruction)")

if __name__ == "__main__":
    main()
