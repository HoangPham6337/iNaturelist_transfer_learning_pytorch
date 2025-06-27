import json

import pandas as pd
from dataset_builder.core import load_config
from pipeline.training import validation_onnx

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

config = load_config("./config.yaml")

species_names = list(species_labels.values())

BATCH_SIZE = 64
NUM_WORKERS = 12

onnx_model_path = "/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.onnx"
report_df = validation_onnx(onnx_model_path, device="cuda")

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    # print(report_df)
    report_df.to_csv("./reports/inceptionv3_50.csv")