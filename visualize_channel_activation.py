import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.dataset_loader import CustomDataset
from pipeline.utility import (
    get_device,
    manifest_generator_wrapper,
    mobile_net_v3_large_builder,
)


def visualize_deadness_map(
    dead_map: torch.Tensor, save_dir: str, layer_name: str, max_channels: int = 64
):
    """
    Visualize per-neuron deadness [C, H, W] as a grid of heatmaps.
    Brighter = more often inactive.

    Args:
        dead_map: Tensor [C, H, W]
        save_dir: Folder to save visualization
        layer_name: Name to use in filename
        max_channels: Max channels to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    C, H, W = dead_map.shape
    C = min(C, max_channels)

    grid_cols = min(8, C)
    grid_rows = math.ceil(C / grid_cols)

    fig, axs = plt.subplots(
        grid_rows, grid_cols, figsize=(2.5 * grid_cols, 2.5 * grid_rows)
    )
    axs = axs.flatten()

    for i in range(C):
        ax = axs[i]
        ax.imshow(dead_map[i], cmap="hot", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"Ch {i}")
        ax.axis("off")

    for i in range(C, len(axs)):
        axs[i].axis("off")  # hide unused plots

    fig.suptitle(f"Deadness Map: {layer_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{layer_name.replace('.', '_')}_deadness_grid.png")
    )
    plt.close()


class NeuronDeadnessTracker:
    def __init__(self, model, threshold=1e-5, target_layer_prefix="features."):
        self.model = model
        self.threshold = threshold
        self.target_layer_prefix = target_layer_prefix
        self.hooks = []
        self.dead_sums = defaultdict(lambda: 0)
        self.total_counts = defaultdict(lambda: 0)
        self._register_hooks()

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            acts = output.detach().abs() < self.threshold
            acts = acts.float().sum(dim=0)
            self.dead_sums[layer_name] += acts.cpu()
            self.total_counts[layer_name] += output.shape[0]

        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name.startswith(self.target_layer_prefix) and isinstance(
                module, torch.nn.Conv2d
            ):
                h = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(h)

    def get_deadness_map(self, layer_name):
        if self.total_counts[layer_name] == 0:
            return None
        return self.dead_sums[layer_name] / self.total_counts[layer_name]


def remove_hooks(self):
    for h in self.hooks:
        h.remove()


device = get_device()
model = mobile_net_v3_large_builder(
    device,
    path="/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.pth",
)
tracker = NeuronDeadnessTracker(model, threshold=0.437038)


output_dir = "activation_sequence"

_, _, val_images, species_labels, _ = manifest_generator_wrapper(0.5)
class_data = []
other_id = list(species_labels.keys())[-1]

val_dataset = CustomDataset(val_images, False)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=15,
    pin_memory=True,
    persistent_workers=True,
)
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        _ = model(images)

# Save raw neuron activations
# for layer in tracker.dead_sums:
#     tensor = tracker.get_deadness_map(layer)
# torch.save(tensor, f"{layer.replace('.', '_')}_neurons.pt")

# Compute deadness
collected_layer = list(tracker.dead_sums.keys())

pruning_plan = {}
# torch.save(dead_map, "features_15_deadness.pt")
all_channel_deadness = []

for layer_name in collected_layer:
    dead_map = tracker.get_deadness_map(layer_name)
    visualize_deadness_map(dead_map, save_dir="deadness_visuals", layer_name=layer_name)  # type: ignore
    if dead_map is None:
        continue
    channel_deadness = dead_map.mean(dim=(1, 2))  # type: ignore
    all_channel_deadness.extend(channel_deadness.tolist())
    pruning_plan[layer_name] = {"channel_deadness": channel_deadness.tolist()}

# Save to JSON
threshold = 25
with open(f"pruning_plan_{threshold}.json", "w") as f:
    json.dump(pruning_plan, f, indent=2)

plt.figure(figsize=(30, 10))
plt.hist(all_channel_deadness, bins=100)
plt.title("Channel Deadness Histogram")
plt.xlabel("Deadness (0 = active, 1 = always dead)")
plt.ylabel("Number of Channels")
plt.savefig(f"channel_deadness_hist_{threshold}.png")
plt.close()

cd_array = np.array(all_channel_deadness)
print("=== Channel Deadness Percentiles ===")
for p in [99.9, 99, 95, 90, 75, 50, 25, 10, 1]:
    print(f"{p}th percentile: {np.percentile(cd_array, p):.3f}")
