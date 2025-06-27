import os
import numpy as np
import onnx
import onnxruntime as ort
from tqdm import tqdm
from torch.utils.data import DataLoader
from pipeline.dataset_loader import CustomDataset
from pipeline.utility import (
    manifest_generator_wrapper,
    get_support_list,
)
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType, QuantFormat, CalibrationDataReader
from typing import Tuple


class ONNXCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.enum_data_dicts = None

    def get_next(self) -> dict:
        if self.enum_data_dicts is None:
            self.enum_data_dicts = self._generate_input()
        return next(self.enum_data_dicts, None)

    def _generate_input(self):
        for images, _ in tqdm(self.dataloader):
            images = images.cpu().numpy().astype(np.float32)
            yield {"input": images}


_, train_images, val_images, species_dict, species_composition = manifest_generator_wrapper(0.5)
val_dataset = CustomDataset(val_images, train=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)
model = onnx.load("/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.onnx")
# input_name = model.graph.input[0].name
# input_shape = model.graph.input[0].type.tensor_type.shape

# print("Input name:", input_name)
# for d in input_shape.dim:
#     print(d.dim_value)

# exit()
quantize_static(
    model_input="/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.onnx",
    model_output="models/mobilenet_v3_large_50_quantized.onnx",
    calibration_data_reader=ONNXCalibrationDataReader(val_loader),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.Entropy
)
