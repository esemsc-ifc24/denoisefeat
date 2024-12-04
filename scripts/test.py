import torch
from diffusers import DDPMPipeline
from utils.feature_extraction import extract_features
from utils.visualizations import visualize_features

def analyze():
    # Load model
    model_name = "google/ddpm-celebahq-256"
    pipeline = DDPMPipeline.from_pretrained(model_name)
    
    # Analyze features
    image = torch.randn((1, 3, 256, 256)).to("mps")  # Random input
    timesteps = torch.arange(0, pipeline.scheduler.num_train_timesteps)
    
    for t in timesteps:
        features = extract_features(pipeline.unet, image, t)
        visualize_features(features, t)

if __name__ == "__main__":
    analyze()
