from datasets import load_dataset

dataset = load_dataset("cifar10", split="train")
print(dataset)
