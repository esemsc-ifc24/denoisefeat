import torch
from torch.optim import Adam
from diffusers import DDPMPipeline
from datasets import load_dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image


def train():
    # Load CIFAR-10 dataset and preprocess images
    dataset = load_dataset("cifar10", split="train")
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])  # Normalize to [-1, 1]

    # Initialize model
    model_name = "google/ddpm-celebahq-256"
    pipeline = DDPMPipeline.from_pretrained(model_name,
                                            torch_dtype=torch.float32).to("cuda")
    optimizer = Adam(pipeline.unet.parameters(), lr=1e-4)

    # Fine-tuning loop
    for epoch in range(5):
        print(f"Epoch {epoch + 1}/5")
        for i, batch in enumerate(dataset):
            # Extract and preprocess image
            image = batch["img"]  # 'img' is the key for CIFAR-10
            if isinstance(image, Image.Image):
                image = transform(image).unsqueeze(0).to("cuda")  # Add batch dimension
            else:
                print(f"Unexpected image type: {type(image)}")
                continue

            # Add noise to the image
            noise = torch.randn_like(image)
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (image.shape[0],)).to("cuda")
            noisy_images = pipeline.scheduler.add_noise(image, noise, timesteps)

            # Forward pass through the UNet model
            outputs = pipeline.unet(noisy_images, timesteps)["sample"]

            # Compute mean squared error loss
            loss = torch.nn.functional.mse_loss(outputs, image)

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Batch {i + 1}: Loss = {loss.item():.4f}")


if __name__ == "__main__":
    train()
