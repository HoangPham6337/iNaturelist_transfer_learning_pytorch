import torch
from typing import Dict
from pipeline.utility import mobile_net_v3_large_builder, get_device

import pprint
import re
from collections import defaultdict


structure: Dict[str, Dict[str, str]]= defaultdict(dict)

device = get_device()
model = mobile_net_v3_large_builder(device, path="/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.pth")

for name, module in model.named_modules():
    block_match = re.match(r"features\.(\d+)\.block\.(.+)", name)
    if not block_match:
        continue
    block_idx, sub_path = block_match.groups()
    block_key = f"features.{block_idx}"

    if sub_path == "0.0":
        structure[block_key]["expand"] = name
    elif sub_path == "1.0":
        structure[block_key]["depthwise"] = name
    elif sub_path == "2.fc1":
        structure[block_key]["se_fc1"] = name
    elif sub_path == "2.fc2":
        structure[block_key]["se_fc2"] = name
    elif sub_path == "3.0":
        structure[block_key]["project"] = name

pprint.pprint(dict(structure))