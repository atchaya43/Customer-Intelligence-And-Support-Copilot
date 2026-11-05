"""Part B: Customer Intelligence, RFM segmentation, KMeans, and Lapse prediction."""

from .rfm_segmentation import perform_segmentation, build_rfm
from .features import build_lapse_dataset, split_lapse_dataset
from .models import train_and_evaluate_lapse_models

__all__ = [
    "perform_segmentation",
    "build_rfm",
    "build_lapse_dataset",
    "split_lapse_dataset",
    "train_and_evaluate_lapse_models",
]
