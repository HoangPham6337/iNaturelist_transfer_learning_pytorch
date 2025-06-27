import json
import random
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import timeit

import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession
from tqdm import tqdm

from pipeline.utility import (
    manifest_generator_wrapper,
    preprocess_eval_opencv,
)


class ModelType(Enum):
    SMALL_MODEL = 0
    BIG_MODEL = 1


class InferenceSpeedBenchmarkFullPipeline:
    def __init__(
        self,
        small_model_path: str,
        big_model_path: str,
        global_data_manifests: List[Tuple[str, int]],
        global_species_labels: Dict[int, str],
        global_total_support_list: Dict[str, int],
        small_model_species_labels: Dict[int, str],
        small_model_input_size: Tuple[int, int] = (224, 224),
        big_model_input_size: Tuple[int, int] = (299, 299),
        is_big_inception_v3: bool = True,
        providers: List[str] = ["CPUExecutionProvider", "CPUExecutionProvider"]
    ) -> None:
        self.small_session = ort.InferenceSession(small_model_path, providers=[providers[0]])
        self.big_session = ort.InferenceSession(big_model_path, providers=[providers[1]])
        self.small_input_name = self.small_session.get_inputs()[0].name
        self.big_input_name = self.big_session.get_inputs()[0].name
        self.small_input_size = small_model_input_size
        self.big_input_size = big_model_input_size

        self.global_data_manifests = global_data_manifests
        self.global_species_labels = global_species_labels
        self.global_total_support_list = global_total_support_list
        self.global_labels_images: Dict[int, List[str]] = defaultdict(list)
        for image_path, species_id in self.global_data_manifests:
            self.global_labels_images[species_id].append(image_path)
        self.global_total_images = sum(len(imgs) for imgs in self.global_labels_images.values())
        self.global_species_probs = {
            int(species_id): len(images) / self.global_total_images
            for species_id, images in self.global_labels_images.items()
        }


        self.small_species_labels: Dict[int, str] = small_model_species_labels
        self.small_other_label = self._get_small_model_other_label()

        self.is_big_incv3 = is_big_inception_v3


    def _get_small_model_other_label(self) -> int:
        species_labels_flip: Dict[str, int] = dict((v, k) for k, v in self.small_species_labels.items())
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


    def _infer_one(self, model_type: ModelType, image_path: str) -> Optional[Tuple[int, float, float]]:
        if model_type == ModelType.BIG_MODEL:
            session: InferenceSession = self.big_session
            input_size = self.big_input_size
            input_name = self.big_input_name
            if self.is_big_incv3:
                is_incv3 = True
            else:
                is_incv3 = False
        else:
            session = self.small_session
            input_size = self.small_input_size
            input_name = self.small_input_name
            is_incv3 = False

        try:
            img = preprocess_eval_opencv(image_path, *input_size, is_inception_v3=is_incv3)
            start = timeit.default_timer()
            outputs = session.run(None, {input_name: img})
            end = timeit.default_timer() - start
            top1_idx = int(np.argmax(outputs[0][0]))
            top1_prob = 1.0
            return top1_idx, top1_prob, end
        except Exception as e:
            print(e)
            return None


    def infer_with_routing(self, image_path: str, ground_truth: int):
        small_result = self._infer_one(ModelType.SMALL_MODEL, image_path)
        if small_result is None:
            print(f"Small model returns no result for {image_path}")
            return None
        small_pred, _, elapsed = small_result

        if small_pred == self.small_other_label:
            big_result = self._infer_one(ModelType.BIG_MODEL, image_path)
            if big_result is None:
                print(f"Big model returns no result for {image_path}")
                return None
            _, _, big_elapsed = big_result
            return ground_truth, 1, elapsed + big_elapsed
        else:
            return ground_truth, 1, elapsed


    def run(
        self,
        num_runs: int,
        sample_size: int = 1000,
    ):
        all_elapsed_times = []
        for run in range(num_runs):
            elapsed_times: List[float] = []
            sampled_species = self._create_stratified_weighted_sample(sample_size)

            for species_id in tqdm(sampled_species, desc=f"Run {run + 1}/{num_runs}", leave=False):
                image_list = self.global_labels_images[int(species_id)]
                if not image_list:
                    print("No image found")
                    continue
                image_path = random.choice(image_list)

                results  = self.infer_with_routing(image_path, species_id)
                if results is not None:
                    _, _, elapsed = results
                    elapsed_times.append(elapsed)
            all_elapsed_times.extend(elapsed_times)
        avg_time_ms = np.mean(all_elapsed_times) * 1000
        std_time_ms = np.std(all_elapsed_times) * 1000
        fps = 1.0 / np.mean(all_elapsed_times)
        print("\nInference Performance:")
        print(f"   Total samples: {len(all_elapsed_times)}")
        print(f"   Average time per image: {avg_time_ms:.2f} ms")
        print(f"   Standard deviation: {std_time_ms:.2f} ms")
        print(f"   Throughput (FPS): {fps:.2f} images/sec")


if __name__ == "__main__":
    threshold = 0.9
    small_image_data, _, _, small_species_labels, _ = manifest_generator_wrapper(threshold)
    global_image_data, _, _, global_species_labels, global_species_composition = manifest_generator_wrapper(1.0)
    with open("./data/haute_garonne/dataset_species_labels_full_bird_insect.json") as full_bird_insect_labels:
        big_species_labels = json.load(full_bird_insect_labels)
        big_species_labels = {int(k) : v for k, v in big_species_labels.items()}

    pipeline = InferenceSpeedBenchmarkFullPipeline(
        f"/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_{int(threshold * 100)}.onnx",
        "/home/tom-maverick/Desktop/convnext_full_inat_bird_insect.onnx",

        global_image_data,
        global_species_labels,
        global_species_composition,
        small_species_labels,
        is_big_inception_v3=False,
        big_model_input_size=(224, 224),
        providers=["CPUExecutionProvider", "CPUExecutionProvider"]
    )
    pipeline.run(1, 1000)