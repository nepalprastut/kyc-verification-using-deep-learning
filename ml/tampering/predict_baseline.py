import argparse
import json
import os
from functools import lru_cache

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import timm
import torch
import torch.nn as nn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
DEFAULT_THRESHOLD = 0.5


def normalize_path(path: str) -> str:
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.normpath(path)


class EfficientNetBinary(nn.Module):
    def __init__(self, model_name="efficientnet_b0", pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


@lru_cache(maxsize=2)
def get_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


@lru_cache(maxsize=4)
def load_model(model_path: str = DEFAULT_MODEL_PATH, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_model_path = normalize_path(model_path)
    model = EfficientNetBinary(model_name="efficientnet_b0", pretrained=False)
    state = torch.load(resolved_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_image(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | None = None,
):
    resolved_image_path = normalize_path(image_path)
    if not os.path.exists(resolved_image_path):
        raise FileNotFoundError(f"Could not read image: {resolved_image_path}")

    model, resolved_device = load_model(model_path=model_path, device=device)

    image = cv2.imread(resolved_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {resolved_image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transforms = get_transforms()
    tensor = transforms(image=image)["image"].unsqueeze(0).to(resolved_device)

    with torch.no_grad():
        logits = model(tensor)
        tampered_probability = float(torch.sigmoid(logits).item())

    tampered_probability = max(0.0, min(1.0, tampered_probability))
    genuine_probability = 1.0 - tampered_probability
    is_tampered = tampered_probability >= threshold

    return {
        "image_path": resolved_image_path,
        "model_path": normalize_path(model_path),
        "device": resolved_device,
        "tampered_probability": round(tampered_probability, 4),
        "genuine_probability": round(genuine_probability, 4),
        "threshold_used": threshold,
        "is_tampered": is_tampered,
        "prediction": "FORGED" if is_tampered else "GENUINE",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON payload instead of human-readable logs",
    )
    args = parser.parse_args()

    result = predict_image(
        image_path=args.image,
        model_path=args.model,
        threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(result))
    else:
        print(f"Image      : {result['image_path']}")
        print(f"Model      : {result['model_path']}")
        print(f"Device     : {result['device']}")
        print(f"Score      : {result['tampered_probability']:.6f}")
        print(f"Threshold  : {result['threshold_used']:.2f}")
        print(f"Prediction : {result['prediction']}")
