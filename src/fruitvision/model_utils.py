from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "keras_model.h5"
LABELS_PATH = ROOT / "models" / "labels.txt"


def load_label_names(labels_path: Path = LABELS_PATH) -> List[str]:
    """Load label names from labels.txt, normalizing the exported truncated names."""
    labels = []
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if " " in line:
                _, label = line.split(" ", 1)
                labels.append(label)
            else:
                labels.append(line)

    if not labels:
        labels = [
            "Manzana_fresca",
            "Manzana_deteriorada",
            "Banano_fresco",
            "Banano_deteriorado",
        ]

    # The exported Teachable Machine labels are truncated in the file; these normalized names
    # are the project-level semantic mapping requested by the assignment.
    normalized = {
        0: "Manzana_fresca",
        1: "Manzana_deteriorada",
        2: "Banano_fresco",
        3: "Banano_deteriorado",
    }
    return [normalized.get(i, labels[i]) for i in range(len(labels))]


def preprocess_image_array(image_array: np.ndarray, mode: str = "mobilenetv2") -> np.ndarray:
    """Prepare image tensor for a MobileNetV2-style model with float32 values and normalization."""
    image = image_array.astype(np.float32)
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] != 3:
        image = image[..., :3]

    image = image / 255.0
    if mode == "mobilenetv2":
        image = image * 2.0 - 1.0
    return image.astype(np.float32)
