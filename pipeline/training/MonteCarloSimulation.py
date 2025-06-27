import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import pandas as pd
import scipy.special
from dataset_builder.core.utility import load_manifest_parquet
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from pipeline.training.utility import (
    create_stratified_weighted_sample,
    false_positive_rate,
    plot_confusion_matrix,
    save_report
)
from pipeline.utility import generate_report, get_support_list, preprocess_eval_opencv


class MonteCarloSimulation:
    """
    A class for running Monte Carlo simulations to evaluate the performance and robustness of an ONNX image classification model.

    The simulation randomly samples species and images, runs model inference, and collects evaluation statistics such as communication rates, false positive rates, and accuracy scores.
    It also supports saving detailed reports and plotting confusion matrices.

    Args:
        model_path (str): 
            Path to the ONNX model to load.
        data_manifest (Union[str, List[Tuple[str, int]]]): 
            Either the path to a Parquet file containing (image_path, label) pairs or a preloaded list of (image_path, label) samples.
        dataset_species_labels (Dict[int, str]): 
            Mapping from integer class IDs to species names.
        input_size (Tuple[int, int], optional): 
            Expected (height, width) for input images. Defaults to (224, 224).
        providers (List[str], optional): 
            List of ONNXRuntime providers for inference (e.g., CPU, CUDA). Defaults to ["CPUExecutionProvider"].

    Notes:
        - The simulation first ensures one sample per species, then fills the rest of the batch using weighted random sampling based on species image availability.
        - The confusion matrix is saved to a file if requested.
        - If `save_path` is provided, a detailed CSV report of classification metrics is generated.
    """
    def __init__(
        self, 
        model_path: str, 
        data_manifest: Union[str, List[Tuple[str, int]]], 
        dataset_species_labels: Dict[int, str], 
        is_inception_v3: bool,
        input_size: Tuple[int, int]=(224, 224),
        providers: List[str]=["CPUExecutionProvider"]
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.species_labels = dataset_species_labels
        
        self.other_class_id = int(self._get_other_id())
        if isinstance(data_manifest, str):
            self.data_manifest = load_manifest_parquet(data_manifest)
        else:
            self.data_manifest = data_manifest
        self.species_to_images = defaultdict(list)
        self.species_probs = {}
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        for image_path, species_id in self.data_manifest:
            self.species_to_images[species_id].append(image_path)

        total_images = sum(len(imgs) for imgs in self.species_to_images.values())
        self.species_probs = {
            int(species_id): len(images) / total_images
            for species_id, images in self.species_to_images.items()
        }
        self.is_inception_v3 = is_inception_v3


    def _get_other_id(self):
        """
        Retrieves the class ID corresponding to the "Other" species label.

        This method inverts the species_labels dictionary (mapping from class ID -> label) to label -> class ID, and attempts to find the class ID associated with "Other".
        If "Other" is not present, returns -1 as a default.

        Returns:
            int: The class ID of the "Other" species if found, otherwise -1.
        """
        species_labels_flip: Dict[str, int] = dict((v, k) for k, v in self.species_labels.items())
        return species_labels_flip.get("Other", -1)


    def _infer_one(self, image_path: str) -> Optional[Tuple[int, float]]:
        """
        Performs model inference on a single input image and returns the top-1 predicted class index along with its probability score.

        Args:
            image_path (str): 
                Path to the input image file.

        Returns:
            Optional[Tuple[int, float]]: 
                A tuple containing:
                    - The predicted class index (int).
                    - The associated top-1 probability score (float).
                Returns None if an error occurs during preprocessing or inference.

        Notes:
            - Images are preprocessed using `preprocess_eval_opencv` to match the model's input size.
            - Softmax is applied to model outputs to obtain probability distributions.
        """
        try:
            img = preprocess_eval_opencv(image_path, *self.input_size, is_inception_v3=self.is_inception_v3)
            outputs = self.session.run(None, {self.input_name: img})
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob
        except Exception as e:
            print(e)
            return None


    def run(
        self,
        species_labels: Dict[int, str],
        species_composition: Dict[str, int],
        num_runs: int=30,
        sample_size: int=1000,
        enable_confusion_matrix: bool=False,
        save_path=None,
    ):
        """
        Runs a Monte Carlo simulation to evaluate model performance by repeatedly sampling 
        and classifying random images across multiple runs.

        For each run, a balanced sample containing at least one image per species is generated. 
        The model's predictions are collected, and key metrics such as communication rate, 
        false positive rate (FPR), and overall classification accuracy are computed.

        Args:
            species_labels (Dict[int, str]): 
                Mapping from species ID to species name.
            species_composition (Dict[str, int]): 
                Dictionary mapping species names to the number of available images.
            num_runs (int, optional): 
                Number of independent simulation runs to perform. Defaults to 30.
            sample_size (int, optional): 
                Total number of images to sample per run. Defaults to 1000.
            plot_confusion_matrix (bool, optional): 
                Whether to generate and save a confusion matrix after simulation. Defaults to False.
            save_path (str, optional): 
                Directory to save a CSV report summarizing classification performance. 
                If None, no report is saved.

        Notes:
            - Each run guarantees at least one sample per species.
            - The remaining images are sampled based on species sampling probabilities.
            - Communication rate is defined as the proportion of predictions labeled as "Other."
            - False positive rate (FPR) is calculated specifically for the "Other" class.
            - If `save_path` is provided, a detailed classification report CSV is saved.
            - If `plot_confusion_matrix` is True, a confusion matrix PNG is generated and saved.

        Output Files (optional):
            - `MonteCarloConfusionMatrix_<model_name>.png`: 
                Saved confusion matrix plot (if enabled).
            - `<model_name>.csv`: 
                Saved classification report containing per-class metrics (if enabled).
        """
        species_names = list(species_labels.values())
        total_support_list = get_support_list(species_composition, species_names)
        comm_rates: List[float] = []
        all_true: List[int] = []
        all_pred: List[int] = []

        for run in range(num_runs):
            y_true: List[int] = []
            y_pred: List[int] = []
            sampled_species = create_stratified_weighted_sample(self.species_labels, self.species_probs, sample_size)
            random.shuffle(sampled_species)

            num_comm = 0
            num_local = 0
            correct = 0

            for species_id in tqdm(sampled_species, desc=f"Run {run + 1}/{num_runs}", leave=False):
                image_list = self.species_to_images[int(species_id)]
                if not image_list:
                    print("No image found")
                    continue
                image_path = random.choice(image_list)
                result = self._infer_one(image_path)
                if result is None:
                    continue
                y_true.append(int(species_id))
                y_pred.append(int(result[0]))
                if result[0] == int(species_id):
                    correct += 1
                top1_idx, top1_prop = result
                if top1_idx == self.other_class_id:
                    num_comm += 1
                else:
                    num_local += 1
            
            total_pred = num_comm + num_local
            comm_rate = num_comm / total_pred if total_pred else 0
            comm_rates.append(comm_rate)
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        model_name = os.path.basename(self.model_path).replace(".onnx", "")
        accuracy = accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred, average="macro")
        if save_path:
            save_report(
                save_path,
                model_name,
                all_true,
                all_pred,
                self.species_labels,
                list(self.species_labels.keys()),
                float(accuracy),
                total_support_list,
                enable_confusion_matrix
            )
        print(f"Accuracy: {accuracy} | Macro F1-Score: {f1:.4f} | FPR: {false_positive_rate(self.other_class_id, all_true, all_pred):.4f}")
        print(f"Average saving rate for {model_name}: {sum(comm_rates)/len(comm_rates):.4f}", end=" ")