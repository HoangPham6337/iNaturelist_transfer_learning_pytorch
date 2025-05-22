import json

block_structure = {
    'features.1': {'depthwise': 'features.1.block.1.0', 'expand': 'features.1.block.0.0'},
    'features.10': {'depthwise': 'features.10.block.1.0', 'expand': 'features.10.block.0.0'},
    'features.11': {'depthwise': 'features.11.block.1.0', 'expand': 'features.11.block.0.0',
                    'project': 'features.11.block.3.0', 'se_fc1': 'features.11.block.2.fc1', 'se_fc2': 'features.11.block.2.fc2'},
    'features.12': {'depthwise': 'features.12.block.1.0', 'expand': 'features.12.block.0.0',
                    'project': 'features.12.block.3.0', 'se_fc1': 'features.12.block.2.fc1', 'se_fc2': 'features.12.block.2.fc2'},
    'features.13': {'depthwise': 'features.13.block.1.0', 'expand': 'features.13.block.0.0',
                    'project': 'features.13.block.3.0', 'se_fc1': 'features.13.block.2.fc1', 'se_fc2': 'features.13.block.2.fc2'},
    'features.14': {'depthwise': 'features.14.block.1.0', 'expand': 'features.14.block.0.0',
                    'project': 'features.14.block.3.0', 'se_fc1': 'features.14.block.2.fc1', 'se_fc2': 'features.14.block.2.fc2'},
    'features.15': {'depthwise': 'features.15.block.1.0', 'expand': 'features.15.block.0.0',
                    'project': 'features.15.block.3.0', 'se_fc1': 'features.15.block.2.fc1', 'se_fc2': 'features.15.block.2.fc2'},
    'features.2': {'depthwise': 'features.2.block.1.0', 'expand': 'features.2.block.0.0'},
    'features.3': {'depthwise': 'features.3.block.1.0', 'expand': 'features.3.block.0.0'},
    'features.4': {'depthwise': 'features.4.block.1.0', 'expand': 'features.4.block.0.0',
                   'project': 'features.4.block.3.0', 'se_fc1': 'features.4.block.2.fc1', 'se_fc2': 'features.4.block.2.fc2'},
    'features.5': {'depthwise': 'features.5.block.1.0', 'expand': 'features.5.block.0.0',
                   'project': 'features.5.block.3.0', 'se_fc1': 'features.5.block.2.fc1', 'se_fc2': 'features.5.block.2.fc2'},
    'features.6': {'depthwise': 'features.6.block.1.0', 'expand': 'features.6.block.0.0',
                   'project': 'features.6.block.3.0', 'se_fc1': 'features.6.block.2.fc1', 'se_fc2': 'features.6.block.2.fc2'},
    'features.7': {'depthwise': 'features.7.block.1.0', 'expand': 'features.7.block.0.0'},
    'features.8': {'depthwise': 'features.8.block.1.0', 'expand': 'features.8.block.0.0'},
    'features.9': {'depthwise': 'features.9.block.1.0', 'expand': 'features.9.block.0.0'}
}

with open("mobilenetv3_block_structure.json", "w") as f:
    json.dump(block_structure, f, indent=2)
