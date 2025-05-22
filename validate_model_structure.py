import torch
from pipeline.utility import mobile_net_v3_large_builder, get_device

def register_forward_hooks(model):
    hooks = []
    def hook_fn(name):
        def hook(module, input, output):
            print(f"[FWD] {name}: {module.__class__.__name__}, input: {input[0].shape}, output: {output.shape}")
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    return hooks


device = get_device()
model = mobile_net_v3_large_builder(device, path="./pruned_mobilnetv3_50_large.pth")
hooks = register_forward_hooks(model)

try:
    out = model(torch.randn(1, 3, 224, 224).to(device))
except RuntimeError as e:
    print(f"❌ Runtime error: {e}")
finally:
    for h in hooks:
        h.remove()