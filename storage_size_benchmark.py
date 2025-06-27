import json
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import timeit

import onnxruntime as ort
import scipy.special
from onnxruntime import InferenceSession
from tqdm import tqdm

from pipeline.utility import (
    manifest_generator_wrapper,
    preprocess_eval_opencv,
)


class InferenceBenchmarkSingleModel:
    def __init__(
        self,
        model_path: str,
        global_data_manifests: List[Tuple[str, int]],
        global_species_labels: Dict[int, str],
        model_species_labels: Dict[int, str],
        model_input_size: Tuple[int, int] = (224, 224),
        is_big_inception_v3: bool = False,
        providers: str = "CPUExecutionProvider"
    ) -> None:
        self.session = ort.InferenceSession(model_path, providers=[providers])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = model_input_size

        self.global_data_manifests = global_data_manifests
        self.global_species_labels = global_species_labels
        self.global_labels_images: Dict[int, List[str]] = defaultdict(list)
        for image_path, species_id in self.global_data_manifests:
            self.global_labels_images[species_id].append(image_path)
        self.global_total_images = sum(len(imgs) for imgs in self.global_labels_images.values())
        self.global_species_probs = {
            int(species_id): len(images) / self.global_total_images
            for species_id, images in self.global_labels_images.items()
        }

        self.is_big_incv3 = is_big_inception_v3
        self.model_species_labels = model_species_labels
        self.other_labels = self._get_other_label()
        self.average_image_size = 232.102  # KB
    
    
    def _get_other_label(self) -> int:
        species_labels_flip: Dict[str, int] = dict((v, k) for k, v in self.model_species_labels.items())
        return species_labels_flip.get("Other", -1)


    def _create_stratified_weighted_sample(self, sample_size: int):
        sampled_species = list(self.global_species_labels.keys())
        remaining_k: int = sample_size - len(sampled_species)
        sampled_species += random.choices(
            population=sampled_species,
            weights=[self.global_species_probs[int(sid)] for sid in self.global_species_labels.keys()],
            k=remaining_k
        )
        random.shuffle(sampled_species)
        return [int(label) for label in sampled_species] 


    def _infer_one(self, image_path: str) -> Optional[Tuple[int, float, float]]:
        session: InferenceSession = self.session
        input_size = self.input_size
        input_name = self.input_name
        if self.is_big_incv3:
            is_incv3 = True
        else:
            is_incv3 = False

        try:
            img = preprocess_eval_opencv(image_path, *input_size, is_inception_v3=is_incv3)
            start = timeit.default_timer()
            outputs = session.run(None, {input_name: img})
            end = timeit.default_timer() - start
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob, end
        except Exception as e:
            print(e)
            return None


    def infer(self, image_path: str, ground_truth: int):
        result = self._infer_one(image_path)
        if result is None:
            print(f"Model returns no result for {image_path}")
            return None

        return result[0], result[2]


    def run(
        self,
        num_runs: int,
        sample_size: int 
    ):
        other_data: List[str] = []
        storage_size = 0
        total = 0
        for run in range(num_runs):
            elapsed_times : List[float] = []
            sampled_species = self._create_stratified_weighted_sample(sample_size)

            for species_id in tqdm(sampled_species, desc=f"Run {run + 1}/{num_runs}", leave=False):
                image_list = self.global_labels_images[int(species_id)]
                if not image_list:
                    print("No image found")
                    continue
                image_path = random.choice(image_list)
                result = self.infer(image_path, species_id)
                if result is not None:
                    pred, elapsed = result
                    elapsed_times.append(elapsed)
                    if pred == self.other_labels:
                        storage_size += self.average_image_size
                        other_data.append(image_path)
                if storage_size >= 1048576:
                    break
                total += 1
        print(f"Total run: {total}")
        print(f"Total image stores: {sum([1 for i in other_data])}")
        print(f"Current storage size: {storage_size}")



if __name__ == "__main__":
    image_data , _, _, species_labels, _ = manifest_generator_wrapper(1.0)
    _, _, _, model_species_labels, _ = manifest_generator_wrapper(0.9)
    
    pipeline = InferenceBenchmarkSingleModel(
        "/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_90.onnx",
        image_data,
        species_labels,
        model_species_labels,
        providers="CUDAExecutionProvider"
    )
    pipeline.run(1, 100000)