import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from pipeline.dataset_loader import CustomDataset
from pipeline.utility import (
    get_device,
    manifest_generator_wrapper,
    mobile_net_v3_large_builder,
)


class ActivationVisualizer:
    def __init__(self, model, device, output_dir="activation_sequence"):
        self.model = model
        self.device = device
        self.feature_maps = {}
        self.layer_info = {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._register_hooks()

    def _save_activation(self, name):
        def hook(model, input, output):
            self.feature_maps[name] = output.detach().cpu()

        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name.startswith("features.") and isinstance(module, torch.nn.Conv2d):
                desc = f"{module.__class__.__name__}_{module.out_channels}ch_k{module.kernel_size[0]}s{module.stride[0]}"
                self.layer_info[name] = desc
                module.register_forward_hook(self._save_activation(name))

    def _overlay_heatmap_on_image(
        self, image_tensor, activation_map, save_path, alpha=0.6, cmap="jet"
    ):
        image_np = to_pil_image(image_tensor).convert("RGB")
        heatmap_np = activation_map.numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (
            heatmap_np.max() - heatmap_np.min() + 1e-6
        )

        plt.figure(figsize=(4, 4))
        plt.imshow(image_np)
        plt.imshow(heatmap_np, cmap=cmap, alpha=alpha)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize(self, image_tensor):
        self.feature_maps.clear()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image_tensor.unsqueeze(0).to(self.device))

        for i, (name, fmap) in enumerate(self.feature_maps.items()):
            fmap_mean = fmap.mean(dim=1)  # [1, H, W]
            upsampled = F.interpolate(
                fmap_mean.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            activation_map = upsampled.squeeze(1)[0]  # [224, 224]

            desc = self.layer_info.get(name, "unknown")
            filename = f"step_{i:02d}_{name.replace('.', '_')}_{desc}.png"
            save_path = os.path.join(self.output_dir, filename)
            self._overlay_heatmap_on_image(
                image_tensor.cpu(), activation_map.cpu(), save_path
            )


device = get_device()
model = mobile_net_v3_large_builder(
    device,
    path="/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_100_baseline.pth",
)

target_layer_names = ["features.6", "features.9", "features.12", "features.15"]

model.eval()
_, _, val_data, species_labels, _ = manifest_generator_wrapper(1.0)
val_dataset = CustomDataset(val_data, False)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=15,
    pin_memory=True,
    persistent_workers=True,
)
visualizer = ActivationVisualizer(model, device)

for images, labels in val_loader:
    visualizer.visualize(images[0])
    break
