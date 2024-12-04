def extract_features(model, x, t):
    features = {}
    hooks = []

    def hook_fn(module, input, output):
        features[module] = output

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn))

    model(x, t)

    for hook in hooks:
        hook.remove()
    
    return features
