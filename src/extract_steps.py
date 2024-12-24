import torch
from models.diffusion import DiffusionModel  # Assume a diffusion model in models/

# Load model
model = DiffusionModel.load_pretrained("path/to/pretrained/model")


# Function to extract intermediate outputs
def extract_steps(input_image, model, save_dir):
    noisy_image = input_image
    steps = model.num_steps
    for step in range(steps):
        noisy_image = model.denoise_step(noisy_image, step)
        save_path = f"{save_dir}/step_{step}.png"
        save_image(noisy_image, save_path)  # Save intermediate image
