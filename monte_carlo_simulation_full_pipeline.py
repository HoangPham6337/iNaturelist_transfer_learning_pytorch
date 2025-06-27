import json

from pipeline.utility import (manifest_generator_wrapper)
from pipeline.training import MonteCarloSimulationFullPipeline


if __name__ == "__main__":
    threshold = 0.9
    small_image_data, _, _, small_species_labels, _ = manifest_generator_wrapper(threshold)
    global_image_data, _, _, global_species_labels, global_species_composition = manifest_generator_wrapper(1.0)
    with open("./data/haute_garonne/dataset_species_labels_full_bird_insect.json") as full_bird_insect_labels:
        big_species_labels = json.load(full_bird_insect_labels)
        big_species_labels = {int(k) : v for k, v in big_species_labels.items()}

    pipeline = MonteCarloSimulationFullPipeline(
        f"/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_{int(threshold * 100)}.onnx",
        # "/home/tom-maverick/Documents/Final Results/InceptionV3_HG_onnx/inceptionv3_full_bird_insect.onnx",
        # "/home/tom-maverick/Documents/Final Results/InceptionV3_HG_onnx/inceptionv3_inat_other_50.onnx",
        "/home/tom-maverick/Documents/Final Results/ConvNeXt/convnext_full_inat_bird_insect.onnx",
        # "/home/tom-maverick/Documents/Final Results/InceptionV3_HG_onnx/inceptionv3_inat_other_50.onnx",

        global_image_data,
        global_species_labels,
        global_species_composition,
        small_species_labels,
        big_species_labels,
        is_big_inception_v3=False,
        big_model_input_size=(224, 224),
        providers=["CUDAExecutionProvider", "CUDAExecutionProvider"]
    )
    pipeline.run(5, 1000, "./baseline_benchmark", True)
