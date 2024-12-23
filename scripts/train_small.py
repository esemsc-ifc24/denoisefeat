from diffusers import DDPMPipeline
import torch
from datasets import load_dataset


def main():
    # Load model
    model_name = "google/ddpm-celebahq-256"
    pipeline = DDPMPipeline.from_pretrained(model_name,
                                            torch_dtype=torch.float32,
                                            use_safetensors=False).to("cuda")
    print("Pipeline loaded successfully.")
    
    # Load dataset
    dataset = load_dataset("cifar10", split="train")
    print("Dataset loaded successfully.")


if __name__ == "__main__":
    main()
