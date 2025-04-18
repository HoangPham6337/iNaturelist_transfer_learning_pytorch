import json
import time

import torch  # type: ignore
import torch_pruning as tp
from torch.utils.data import DataLoader
from pipeline.dataset_loader import CustomDataset
from pipeline.training import save_model, train_one_epoch, train_validate, sparse_warmup_epoch
from pipeline.utility import manifest_generator_wrapper
from pipeline.utility.utility import get_device, mobile_net_v3_large_builder

manifest_generator_wrapper()
print()
device = get_device()
print()

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 50
NUM_SPECIES = len(species_labels.keys())
NAME = "mobilenet_v3_large_90_optimized_pruning"

model = mobile_net_v3_large_builder(device, path="./models/mobilenet_v3_large_90.pth")
train_dataset = CustomDataset("./data/haute_garonne/train.parquet", train=True)
val_dataset = CustomDataset("./data/haute_garonne/val.parquet", train=False)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

example_inputs = torch.randn(1, 3, 224, 224).to(device)
ignored_layers = [model.classifier[3]]
importance = tp.importance.BNScaleImportance()
pruner = tp.pruner.BNScalePruner(
    model=model,
    example_inputs=example_inputs,
    importance=importance,
    global_pruning=True,
    isomorphic=True,
    pruning_ratio=0.3,
    ignored_layers=ignored_layers,
    round_to=8
)

warmup_epochs = 3

for epoch in range(warmup_epochs):
    train_loss, train_acc = sparse_warmup_epoch(model, train_loader, criterion, optimizer, pruner, device)
    print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

tp.utils.print_tool.before_pruning(model)
pruner.step()
tp.utils.print_tool.after_pruning(model)
save_model(model, f"{NAME}", "models", device, (224, 224))