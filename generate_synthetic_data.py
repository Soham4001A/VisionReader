import torch
import os
import torchvision.utils as vutils
from PIL import Image

# Import the VAE class from the trainer script
from a_2_vae_trainer import VAE 

# --- Configuration ---
MODEL_PATH = "models/vae.pth"
OUTPUT_DIR = "synthetic_data"
NUM_IMAGES_TO_GENERATE = 5000
LATENT_DIM = 32 # Must match the trained VAE
IMAGE_SIZE = 64 # Must match the trained VAE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please run '2_vae_trainer.py' first.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Load the trained VAE model
    model = VAE(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode

    print(f"--- Generating {NUM_IMAGES_TO_GENERATE} synthetic images ---")
    with torch.no_grad():
        for i in range(NUM_IMAGES_TO_GENERATE):
            # Sample from the latent space (a standard normal distribution)
            z = torch.randn(1, LATENT_DIM).to(device)
            
            # Decode the latent vector into an image
            generated_image = model.decode(model.decoder_fc(z)).cpu()
            
            # Save the image
            file_path = os.path.join(OUTPUT_DIR, f'synthetic_{i+1}.png')
            vutils.save_image(generated_image, file_path, normalize=True)
            
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{NUM_IMAGES_TO_GENERATE} images...")

    print("\n--- Synthetic Data Generation Finished ---")
    print(f"✅ {NUM_IMAGES_TO_GENERATE} images saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()