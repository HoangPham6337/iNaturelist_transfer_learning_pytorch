import torch
import os
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from pipeline.dataset_loader import CustomDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from pipeline.utility import mobile_net_v3_large_builder, get_device, manifest_generator_wrapper
from typing import Dict, List
import numpy as np

class NeuronActivationStatsTracker:
    def __init__(self, model, target_layer_prefix="features."):
        self.model = model
        self.target_layer_prefix = target_layer_prefix
        self.hooks = []
        self.activations = defaultdict(list)
        self._register_hooks()

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            # Collect all activation values (absolute), flatten per sample
            abs_activations = output.detach().abs().cpu().flatten()
            self.activations[layer_name].append(abs_activations)
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name.startswith(self.target_layer_prefix) and isinstance(module, torch.nn.Conv2d):
                h = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def plot_histograms(self, save_dir="histograms", bins=100):
        os.makedirs(save_dir, exist_ok=True)
        for layer_name, acts_list in self.activations.items():
            flat_acts = torch.cat(acts_list).numpy()
            plt.hist(flat_acts, bins=bins, log=True)
            plt.title(f"Activation Histogram - {layer_name}")
            plt.xlabel("Activation Magnitude")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(save_dir, f"{layer_name.replace('.', '_')}_hist.png"))
            plt.close()


    def plot_histogram(self):
        all_acts = []
        for acts_list in tracker.activations.values():
            all_acts.append(torch.cat(acts_list))

        flat_all_acts = torch.cat(all_acts).numpy()
        plt.figure(figsize=(30, 10))
        plt.hist(flat_all_acts, bins=10000, log=True)
        plt.title("All Layer Activations")
        plt.xlabel("Activation Magnitude")
        plt.ylabel("Frequency")
        plt.savefig("activation_histograms/all_layers_combined_hist.png")
        plt.close()


    def analyze(self):
        if not self.activations:
            print("No activations collected yet.")
            return

        all_acts = []
        for acts_list in self.activations.values():
            all_acts.append(torch.cat(acts_list))  # each acts_list = list of flattened tensors

        flat_acts = torch.cat(all_acts).numpy()

        print("=== Activation Magnitude Percentiles ===")
        print(f"Max:               {np.max(flat_acts):.6f}")
        print(f"99.9th percentile: {np.percentile(flat_acts, 99.9):.6f}")
        print(f"99th percentile:   {np.percentile(flat_acts, 99):.6f}")
        print(f"90th percentile:   {np.percentile(flat_acts, 90):.6f}")
        print(f"75th percentile:   {np.percentile(flat_acts, 75):.6f}")
        print(f"50th percentile:   {np.percentile(flat_acts, 50):.6f}")
        print(f"25th percentile:   {np.percentile(flat_acts, 25):.6f}")
        print(f"10th percentile:   {np.percentile(flat_acts, 10):.6f}")
        print(f"1st percentile:    {np.percentile(flat_acts, 1):.6f}")
        print(f"Min:               {np.min(flat_acts):.6f}")


def remove_hooks(self):
    for h in self.hooks:
        h.remove()

device = get_device()
model = mobile_net_v3_large_builder(device, path="/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.pth")
tracker = NeuronActivationStatsTracker(model)


output_dir = "activation_sequence"

_ , _, val_images, species_labels, _ = manifest_generator_wrapper(0.5)

species_to_images: Dict[int, List[str]] = defaultdict(list)
species_probs = {}
sample_size = 500

for image_path, species_id in val_images:
    species_to_images[species_id].append(image_path)

total_images = sum(len(imgs) for imgs in species_to_images.values())
species_probs = {
    int(species_id): len(images) / total_images
    for species_id, images in species_to_images.items()
}
sampled_species = []
for species in species_labels.keys():
    sampled_species.append(species)

remaining_k = sample_size - len(sampled_species)
sampled_species += random.choices(
    population=list(species_labels.keys()),
    weights=[species_probs[int(sid)] for sid in species_labels.keys()],
    k=remaining_k
)
final_data = []
for species_id in sampled_species:
    image_list = species_to_images[int(species_id)]
    if not image_list:
        print("No image found")
        continue
    image_path = random.choice(image_list)
    final_data.append((image_path, species_id))
val_dataset = CustomDataset(final_data, False)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=15,
    pin_memory=True,
    persistent_workers=True
)
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        _ = model(images)

tracker.plot_histograms(save_dir="activation_histograms", bins=100)
tracker.analyze()
