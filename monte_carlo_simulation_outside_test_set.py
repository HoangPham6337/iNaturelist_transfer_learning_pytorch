import json


from pipeline.utility import (
    manifest_generator_wrapper,
)

from pipeline.training import MonteCarloSimulationOutsideTestSet


if __name__ == "__main__":
    test_image_data, _, _, test_species_labels, test_species_composition = (
        manifest_generator_wrapper(1.0)
    )
    with open(
        "./data/haute_garonne/dataset_species_labels_full_bird_insect.json"
    ) as full_bird_insect_labels:
        big_species_labels = json.load(full_bird_insect_labels)
        big_species_labels = {int(k): v for k, v in big_species_labels.items()}

    pipeline = MonteCarloSimulationOutsideTestSet(
        "/home/tom-maverick/Documents/Final Results/ConvNeXt/convnext_full_inat_bird_insect.onnx",
        test_image_data,
        test_species_labels,
        test_species_composition,
        big_species_labels,
        providers=["CUDAExecutionProvider"],
    )
    pipeline.run(1, 1000, False, "./baseline_benchmark")