import torch
from diffusers import UNet2DConditionModel


class DiffusionModel:
    def __init__(self, model_name="google/ddpm-celebahq-256"):
        self.model = UNet2DConditionModel.from_pretrained(model_name)

    def forward(self, x, timesteps, **kwargs):
        return self.model(x, timesteps, **kwargs)

    def extract_features(self, x, timesteps):

        features = {}
        hooks = []
        
        def hook_fn(module, input, output):
            features[module] = output

        for name, module in self.model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn))
        
        self.model(x, timesteps)
        
        for hook in hooks:
            hook.remove()
        
        return features
