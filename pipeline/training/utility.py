import os
import random
from typing import List, Dict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pipeline.utility import generate_report
import pandas as pd
import matplotlib.pyplot as plt


def create_stratified_weighted_sample(
    species_labels: Dict[int, str], species_probs: Dict[int, float], sample_size: int
):
    sampled_species = list(species_labels.keys())
    remaining_k: int = sample_size - len(sampled_species)
    sampled_species += random.choices(
        population=sampled_species,
        weights=[species_probs[int(sid)] for sid in species_labels.keys()],
        k=remaining_k,
    )
    random.shuffle(sampled_species)
    return [int(label) for label in sampled_species]


def false_positive_rate(other_class_id: int, y_true: List[int], y_pred: List[int]):
    """
    Computes the false positive rate (FPR) for the "Other" class predictions.

    Specifically, it measures how often the model incorrectly predicts a sample that should be classified as "Other" (communication) into a different (local) class.

    Args:
        y_true (List[int]):
            Ground truth class IDs.
        y_pred (List[int]):
            Predicted class IDs from the model.

    Returns:
        float:
            The false positive rate, calculated as:
                FPR = False Positives / (False Positives + True Negatives)
            Returns 0.0 if there are no samples of the "Other" class.

    Notes:
        - "False Positive" (FP):
            True label is "Other", but the model predicts a different class.
        - "True Negative" (TN):
            True label is "Other", and the model correctly predicts "Other".
        - If there are no "Other" class samples, the function returns 0.0.
    """
    # true: local / false: communication
    # fp: it should be communicate but model predict as local
    # tn: it should be communicate, model predict as communicate
    fp_count = 0
    tn_count = 0
    for label, predict in zip(y_true, y_pred):
        if label == other_class_id:
            if predict != label:
                fp_count += 1
            else:
                tn_count += 1
    if (fp_count + tn_count) == 0:
        return 0.0
    return fp_count / (fp_count + tn_count)


def plot_confusion_matrix(
    save_path: str, species_labels: Dict[int, str], y_true: List[int], y_pred: List[int]
):
    """
    Plots and saves a confusion matrix for the model's predictions on the given true labels.

    Args:
        y_true (List[int]):
            Ground truth class IDs.
        y_pred (List[int]):
            Predicted class IDs from the model.

    Notes:
        - Uses `sklearn.metrics.ConfusionMatrixDisplay` to visualize the matrix.
        - Axis labels are derived from `self.species_labels` (class ID to name).
        - The plot is saved as a PNG file with the filename:
            "MonteCarloConfusionMatrix_<model_name>.png"
        - The figure size is large (40x40 inches).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(map(int, species_labels.keys())))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(species_labels.values())
    )
    fig, ax = plt.subplots(figsize=(40, 40))
    disp.plot(
        ax=ax,
        xticks_rotation=50,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )
    plt.title("Confusion Matrix (Monte Carlo Simulation)", fontsize=40)
    plt.tight_layout()
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=25)
    cbar_ax.set_ylabel("Count", fontsize=30)
    plt.savefig(os.path.join(save_path, "MonteCarloConfusionMatrix.png"))
    plt.close(fig)


def get_other_id(species_labels: Dict[int, str]) -> int:
    species_labels_flip: Dict[str, int] = dict(
        (v, k) for k, v in species_labels.items()
    )
    return species_labels_flip.get("Other", -1)


def save_report(
    save_path: str,
    model_name: str,
    all_true: List[int],
    all_pred: List[int],
    species_labels: Dict[int, str],
    unique_ids: List[int],
    accuracy: float,
    support_list: List[int],
    enable_confusion_matrix: bool=False
):
    df = generate_report(
        all_true,
        all_pred,
        list(species_labels.values()),
        unique_ids,
        support_list,
        float(accuracy),
    )

    os.makedirs(save_path, exist_ok=True)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        df.to_csv(os.path.join(save_path, f"{model_name}.csv"))
    
    if enable_confusion_matrix:
        plot_confusion_matrix(save_path, species_labels, all_true, all_pred)