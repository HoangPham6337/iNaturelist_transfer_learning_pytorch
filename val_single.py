import json
import pandas as pd
import torch  # type: ignore
import numpy as np
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from pipeline.utility.utility import get_device, get_support_list, generate_report, mobile_net_v3_large_builder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)
from pipeline.dataset_loader import CustomDataset
from pipeline.utility import manifest_generator_wrapper


device = get_device()
BATCH_SIZE = 64
NUM_WORKERS = 12
NAME = "mobilenet_v3_large"

manifest_generator_wrapper()

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

species_names = list(species_labels.values())
val_dataset = CustomDataset("./data/haute_garonne/val.parquet", train=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

total_support_list = get_support_list("./data/haute_garonne/species_composition.json", species_names)

model = mobile_net_v3_large_builder(device, path="/home/tom-maverick/Desktop/mobilenet_v3_large.pth")

model.eval()
val_loss, val_correct = 0.0, 0

all_preds: List[np.ndarray] = []
all_labels: List[np.ndarray] = []

print("Begin validating")
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validating", unit="Batch"):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        # print("Prediction counts:", np.unique(all_preds, return_counts=True))
        all_labels.extend(labels.cpu().numpy())
        # print("Label counts:", np.unique(all_labels, return_counts=True))

accuracy = accuracy_score(all_labels, all_preds)
weighted_recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")

print(f"Validation accuracy: {accuracy:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted Average F1-Score: {f1:.4f}")
report_df = generate_report(all_labels, all_preds, species_names, total_support_list, float(accuracy))

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    # print(report_df)
    report_df.to_csv("./reports/mobilenet_v3_large_50.csv")