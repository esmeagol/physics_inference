"""
ReplayDetector: InferenceRunner that replays detections from a JSONL file.

Each line of the file should be a JSON object with keys:
- frame: int
- time: float (optional)
- predictions: list of detection dicts with x, y, width, height, confidence, class, class_id
- image: { width, height } (optional)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .inference_runner import InferenceRunner


class ReplayDetector(InferenceRunner):
    def __init__(self, jsonl_path: str, loop: bool = False) -> None:
        self.jsonl_path = jsonl_path
        self.loop = loop
        self.records: List[Dict[str, Any]] = []
        self._cursor: int = 0
        self._load()

    def _load(self) -> None:
        try:
            with open(self.jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.records.append(json.loads(line))
                    except Exception:
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"ReplayDetector JSONL not found: {self.jsonl_path}")

        if not self.records:
            raise ValueError(f"ReplayDetector empty JSONL: {self.jsonl_path}")

    def predict(self, image: Union[str, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        if self._cursor >= len(self.records):
            if self.loop:
                self._cursor = 0
            else:
                # Return empty predictions at EOF
                return {"predictions": [], "image": {}}

        rec = self.records[self._cursor]
        self._cursor += 1

        # Ensure keys exist
        preds = rec.get("predictions", [])
        image_info = rec.get("image", {})
        model = rec.get("model", "ReplayDetector")
        return {
            "predictions": preds,
            "image": image_info,
            "model": model,
        }

    def predict_batch(self, images: List[Union[str, np.ndarray]], **kwargs: Any) -> List[Dict[str, Any]]:
        return [self.predict(img, **kwargs) for img in images]

    def visualize_predictions(self, image: Union[str, np.ndarray], predictions: Dict, output_path: Optional[str] = None) -> np.ndarray:
        # No special visualization; rely on upstream visualization helpers
        if isinstance(image, str):
            raise ValueError("ReplayDetector.visualize_predictions expects a numpy image array")
        return image

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "ReplayDetector",
            "type": "replay",
            "path": self.jsonl_path,
            "size": len(self.records),
        }



