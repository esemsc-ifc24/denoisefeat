import torch
from diffusers import DDPMPipeline
from datasets import load_dataset

def train():
    # Load dataset
    dataset = load_dataset("huggan/flowers-102", split="train")
    
    # Initialize model
    model_name = "google/ddpm-celebahq-256"
    pipeline = DDPMPipeline.from_pretrained(model_name)

    # Fine-tuning loop (if needed)
    for epoch in range(5):
        for batch in dataset:
            images = batch["image"].to("mps")  # Adapt for Mac M2
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (images.shape[0],))
            
            # Denoising process
            noisy_images = pipeline.scheduler.add_noise(images, noise, timesteps)
            outputs = pipeline.unet(noisy_images, timesteps)
            loss = compute_loss(outputs, images)
            
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()
