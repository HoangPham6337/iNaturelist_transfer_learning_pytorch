import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import scipy.special
from onnxruntime import InferenceSession
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from pipeline.utility import (
    generate_report,
    get_support_list,
    manifest_generator_wrapper,
    preprocess_eval_opencv,
)

from pipeline.training.utility import create_stratified_weighted_sample, save_report


class MonteCarloSimulationOutsideTestSet:
    def __init__(
        self,
        model_path: str,
        test_data_manifests: List[Tuple[str, int]],
        test_species_labels: Dict[int, str],
        test_total_support_list: Dict[str, int],
        model_species_labels: Dict[int, str],
        model_input_size: Tuple[int, int] = (224, 224),
        providers: List[str] = ["CPUExecutionProvider"],
    ) -> None:
        self.model_name = os.path.basename(model_path).replace(".onnx", "")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = model_input_size

        self.test_data_manifests = test_data_manifests
        self.test_species_labels = test_species_labels
        self.test_species_names = list(self.test_species_labels.values())
        self.test_total_support_list = test_total_support_list
        self.test_labels_images: Dict[int, List[str]] = defaultdict(list)
        for image_path, species_id in self.test_data_manifests:
            self.test_labels_images[species_id].append(image_path)
        self.test_total_images = sum(
            len(imgs) for imgs in self.test_labels_images.values()
        )
        self.test_species_probs = {
            int(species_id): len(images) / self.test_total_images
            for species_id, images in self.test_labels_images.items()
        }

        self.not_belong_to_global_idx = len(self.test_species_labels)

        self.model_species_labels: Dict[int, str] = model_species_labels
        self.species_name = list(self.model_species_labels.values())

    def _is_prediction_belongs_to_test_dataset(self, prediction: int) -> bool:
        species_name: str | None = self.model_species_labels.get(prediction, None)

        if species_name is None:
            print(
                f"[Warning] Species name not in model model species labels: {prediction}"
            )
            return False

        if species_name not in self.test_species_names:
            return False

        return True

    def _translate_prediction_to_test_label(self, prediction: int):
        model_species_labels = self.model_species_labels.get(prediction, None)
        global_species_labels = list(
            filter(
                lambda key: self.test_species_labels[key] == model_species_labels,
                self.test_species_labels,
            )
        )
        if not global_species_labels:
            print(
                f"[Warning] Could not map species from model prediction {model_species_labels} to test label"
            )
            return self.not_belong_to_global_idx
        return global_species_labels[0]

    def _create_stratified_weighted_sample(self, sample_size: int):
        sampled_species = list(self.test_species_labels.keys())
        remaining_k: int = sample_size - len(sampled_species)
        sampled_species += random.choices(
            population=sampled_species,
            weights=[
                self.test_species_probs[int(sid)]
                for sid in self.test_species_labels.keys()
            ],
            k=remaining_k,
        )
        random.shuffle(sampled_species)
        return [int(label) for label in sampled_species]

    def _infer_one(self, image_path: str) -> Optional[Tuple[int, float]]:
        session: InferenceSession = self.session
        input_size = self.input_size
        input_name = self.input_name

        try:
            img = preprocess_eval_opencv(image_path, *input_size)
            outputs = session.run(None, {input_name: img})
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob
        except Exception as e:
            print(e)
            return None

    def infer_with_routing(self, image_path: str, ground_truth: int):
        result = self._infer_one(image_path)
        if result is None:
            print(f"Big model returns no result for {image_path}")
            return None
        if not self._is_prediction_belongs_to_test_dataset(result[0]):
            return ground_truth, self.not_belong_to_global_idx

        big_species_name = self.model_species_labels.get(result[0], None)
        if big_species_name is None:
            print(f"Failed to get species name for big label: {result[0]}")
            return None
        global_pred = self._translate_prediction_to_test_label(result[0])
        return ground_truth, global_pred

    def run(
        self,
        num_runs: int,
        sample_size: int = 1000,
        enable_confusion_matrix: bool = False,
        save_path=None,
    ):
        all_true, all_pred = [], []
        for run in range(num_runs):
            y_true: List[int] = []
            y_pred: List[int] = []
            sampled_species = create_stratified_weighted_sample(
                self.test_species_labels, self.test_species_probs, sample_size
            )

            for species_id in tqdm(
                sampled_species, desc=f"Run {run + 1}/{num_runs}", leave=False
            ):
                image_list = self.test_labels_images[int(species_id)]
                if not image_list:
                    print("No image found")
                    continue
                image_path = random.choice(image_list)
                result = self.infer_with_routing(image_path, species_id)
                if result is not None:
                    ground_truth, pred = result
                    y_true.append(ground_truth)
                    y_pred.append(pred)
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        accuracy = accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred, average="macro")

        total_support_list = get_support_list(self.test_total_support_list, self.test_species_names)
        num_pred_outside_global = sum([1 for pred in all_pred if pred == self.not_belong_to_global_idx])
        self.test_species_labels.update({self.not_belong_to_global_idx: "Not in HG dataset"})
        total_support_list.append(num_pred_outside_global)
        unique_ids = list(self.test_species_labels.keys())

        print(f"Accuracy: {accuracy} | Macro F1-Score: {f1:.4f}")
        print(f"Total prediction outside the test set: {num_pred_outside_global}")
        print(f"Test sample size across all run: {len(all_true)}")

        if save_path:
            save_report(
                save_path,
                self.model_name,
                all_true,
                all_pred,
                self.test_species_labels,
                unique_ids,
                float(accuracy),
                total_support_list,
                enable_confusion_matrix,
            )