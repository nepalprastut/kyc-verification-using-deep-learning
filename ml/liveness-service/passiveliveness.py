# -*- coding: utf-8 -*-
"""Passive liveness inference helpers and CLI using final_model.pt."""

import argparse
import json
import os
from collections import OrderedDict
from functools import lru_cache

from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# ------------------ Config ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "final_model.pt")
IMG_SIZE = 224
CLASS_NAMES = ["live", "spoof"]
DEFAULT_THRESHOLD = 0.5
COMMON_CHECKPOINT_KEYS = ("state_dict", "model_state_dict", "model", "net")

# Must match training code
DROPOUT_RATE = 0.3
DENSE_UNITS = 128


# ------------------ Exact trained architecture ------------------
class MobileNetV2Liveness(nn.Module):
    def __init__(self, dropout_rate: float = DROPOUT_RATE, dense_units: int = DENSE_UNITS):
        super().__init__()

        # Same backbone family used in training
        self.backbone = models.mobilenet_v2(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1)
        )

    def forward(self, x):
        return self.backbone(x)


# ------------------ Helpers ------------------
def normalize_path(path: str) -> str:
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.normpath(path)


def _is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except UnicodeDecodeError:
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1"


def _load_checkpoint(model_path: str):
    try:
        return torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=torch.device("cpu"))


def _extract_state_dict(checkpoint):
    # Raw state_dict
    if isinstance(checkpoint, OrderedDict):
        return checkpoint

    # Checkpoint dict
    if isinstance(checkpoint, dict):
        for key in COMMON_CHECKPOINT_KEYS:
            candidate = checkpoint.get(key)
            if isinstance(candidate, OrderedDict):
                return candidate

    return None


def _clean_state_dict_keys(state_dict: OrderedDict) -> OrderedDict:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = value
    return cleaned


def _build_exact_model() -> nn.Module:
    model = MobileNetV2Liveness()
    model.eval()
    return model


@lru_cache(maxsize=4)
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    resolved_path = normalize_path(model_path)

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Model not found: {resolved_path}")

    if _is_git_lfs_pointer(resolved_path):
        raise RuntimeError(
            "Model file is a Git LFS pointer, not the actual weights. "
            "Run `git lfs pull` or replace the pointer file with the real model file."
        )

    checkpoint = _load_checkpoint(resolved_path)

    # If someone saved the whole module
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint

    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise RuntimeError(
            f"Unsupported passive liveness model format: {type(checkpoint).__name__}"
        )

    state_dict = _clean_state_dict_keys(state_dict)
    model = _build_exact_model()

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        first_keys = list(state_dict.keys())[:20]
        raise RuntimeError(
            "Failed to load final_model.pt into MobileNetV2Liveness.\n"
            f"First checkpoint keys: {first_keys}\n"
            f"Original error: {exc}"
        ) from exc

    model.eval()
    return model


def preprocess_pil_image(pil_img, img_size: int = IMG_SIZE):
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return preprocess(pil_img).unsqueeze(0)


# ------------------ Prediction ------------------
def predict_image(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
):
    resolved_image_path = normalize_path(image_path)

    if not os.path.exists(resolved_image_path):
        raise FileNotFoundError(f"Image not found: {resolved_image_path}")

    model = load_model(model_path)

    pil_img = Image.open(resolved_image_path).convert("RGB")
    x = preprocess_pil_image(pil_img)

    with torch.no_grad():
        output = model(x)

        # Trained model outputs a single logit
        if output.ndim == 2 and output.shape[1] == 1:
            spoof_probability = torch.sigmoid(output[0, 0]).item()
        elif output.ndim == 1 and output.shape[0] == 1:
            spoof_probability = torch.sigmoid(output[0]).item()
        else:
            # fallback only if a 2-class head is ever used later
            probs = torch.softmax(output, dim=1)
            spoof_probability = probs[0, 1].item()

    spoof_probability = max(0.0, min(1.0, spoof_probability))
    pred_idx = 1 if spoof_probability >= threshold else 0
    pred_label = CLASS_NAMES[pred_idx]
    confidence = spoof_probability if pred_idx == 1 else (1.0 - spoof_probability)
    live_probability = 1.0 - spoof_probability

    return {
        "image_path": resolved_image_path,
        "model_path": normalize_path(model_path),
        "prediction": pred_label,
        "passed": pred_label == "live",
        "spoof_probability": round(spoof_probability, 4),
        "live_probability": round(live_probability, 4),
        "confidence": round(confidence, 4),
        "threshold_used": float(threshold),
    }


# ------------------ CLI ------------------
def main():
    parser = argparse.ArgumentParser(description="Face passive liveness detection")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to final_model.pt",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input face image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Spoof threshold",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output",
    )
    args = parser.parse_args()

    result = predict_image(
        image_path=args.image,
        model_path=args.model,
        threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("-" * 40)
    print(f"Image:             {os.path.basename(result['image_path'])}")
    print(f"Prediction:        {result['prediction'].upper()}")
    print(f"Spoof probability: {result['spoof_probability']:.4f}")
    print(f"Live probability:  {result['live_probability']:.4f}")
    print(f"Confidence:        {result['confidence']:.4f}")
    print(f"Threshold:         {result['threshold_used']:.2f}")
    print("-" * 40)


if __name__ == "__main__":
    main()