import json
import torch
from typing import Dict, List
import torch_pruning as tp
from pipeline.utility import get_device, mobile_net_v3_large_builder

with open("./pruning_plan_25.json") as f1:
    layer_prune_data: Dict[str, Dict[str, List[int]]] = json.load(f1)


with open("./mobilenetv3_block_structure.json") as f2:
    block_structure: Dict[str, Dict[str, str]] = json.load(f2)


block_pruning_plan = {}

deadness_threshold = 0.12

for block_name, layers in block_structure.items():
    expand_name = layers.get("expand")

    if not expand_name:
        print("No expand name")
        continue

    if expand_name not in layer_prune_data:
        continue

    deadness_scores = layer_prune_data[expand_name]["channel_deadness"]
    prune_idxs = [i for i, score in enumerate(deadness_scores) if score > deadness_threshold]

    if not prune_idxs:
        print(f"Nothing to prune in {block_name}")
        continue

    block_pruning_plan[block_name] = {
        "prune_channels": prune_idxs,
        "channel_deadness": deadness_scores
    }

device = get_device()
model = mobile_net_v3_large_builder(device, path="/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.pth")

example_input = torch.randn(1, 3, 224, 224).to(device)
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=example_input)
named_modules = dict(model.named_modules())

for block_name, plan in block_pruning_plan.items():
    prune_idxs = plan["prune_channels"]
    layers = block_structure[block_name]

    pruning_groups = []
    layer = named_modules[layers["depthwise"]]
    if not (layer.in_channels == layer.out_channels == layer.groups):
        print(f"[WARNING] {block_name}: depthwise structure invalid before pruning")
        continue

    try:
        original_channels = layer_prune_data[layers["expand"]]["channel_deadness"]
        min_keep_ratio = 0.25
        min_channels_to_keep = max(8, int(len(original_channels) * min_keep_ratio))

        max_prune = len(original_channels) - min_channels_to_keep
        if len(prune_idxs) > max_prune:
            prune_idxs = prune_idxs[:max_prune]
        idxs = prune_idxs  # based on expand layer


        # Expand
        pruning_groups.append(DG.get_pruning_group(named_modules[layers["expand"]], tp.prune_conv_out_channels, idxs=idxs))

        # Depthwise — safe if diagonal
        depthwise = named_modules[layers["depthwise"]]
        if depthwise.groups == depthwise.in_channels == depthwise.out_channels:
            pruning_groups.append(DG.get_pruning_group(depthwise, tp.prune_conv_in_channels, idxs=idxs))
            pruning_groups.append(DG.get_pruning_group(depthwise, tp.prune_conv_out_channels, idxs=idxs))

        # SE
        if "se_fc1" in layers:
            se1 = named_modules[layers["se_fc1"]]
            pruning_groups.append(DG.get_pruning_group(se1, tp.prune_conv_in_channels, idxs=idxs))  # or linear
        if "se_fc2" in layers:
            se2 = named_modules[layers["se_fc2"]]
            pruning_groups.append(DG.get_pruning_group(se2, tp.prune_conv_in_channels, idxs=idxs))

        # Project
        if "project" in layers:
            pruning_groups.append(DG.get_pruning_group(named_modules[layers["project"]], tp.prune_conv_in_channels, idxs=idxs))

        if all(DG.check_pruning_group(pg) for pg in pruning_groups):
            for pg in pruning_groups:
                pg.prune()
            print(f"Pruned {block_name}: {len(prune_idxs)} channels")
        else:
            print(f"[SKIP] {block_name}: unsafe pruning group")
            layer_names = [
                layers["expand"],
                layers["depthwise"],
                layers["depthwise"],  # reused for in and out
                layers.get("se_fc1"),
                layers.get("se_fc2"),
                layers.get("project")
            ]

            for i, pg in enumerate(pruning_groups):
                layer_name = layer_names[i] if i < len(layer_names) else f"unknown_layer_{i}"
                if not DG.check_pruning_group(pg):
                    print(f"  ↳ Group {i} is invalid (layer: {layer_name})")
    except Exception as e:
        print(f"[ERROR] {block_name}: {e}")

torch.save(model, "./pruned_mobilnetv3_50_large.pth")