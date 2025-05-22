import re
import json
from collections import defaultdict

block_channel_deadness = defaultdict(list)
with open("./pruning_plan_10.json") as pruning_data:
    pruning_plan = json.load(pruning_data)

for layer_name, stats in pruning_plan.items():
    # Extract block name (e.g., "features.1") from "features.1.block.0.0"
    match = re.match(r"(features\.\d+)\.", layer_name)
    if not match:
        continue  # skip non-block layers

    block_name = match.group(1)
    block_channel_deadness[block_name].extend(stats["channel_deadness"])
with open("pruning_plan_block.json", "w") as f:
    json.dump(block_channel_deadness, f, indent=2)