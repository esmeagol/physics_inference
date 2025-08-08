"""
SnookerBallDetector Module

Implements a rewritten predict() pipeline for combining multiple detectors
and resolving detections into a consistent snooker table state using a
6-step algorithm:

1) Run each detector, normalize predictions, and collect per-detector results
2) Estimate expected ball size statistics (median, p10, p90)
3) Spatially group detections across detectors (within 20% of median size)
4) Resolve high-certainty objects (multi-detector, single-ball-sized, stable wrt previous frame)
5) Resolve colored balls with confidence threshold
6) Resolve remaining groups to expected balls (including red splitting and ambiguity handling)
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .inference_runner import InferenceRunner


class SnookerBallDetector(InferenceRunner):
    """
    Ensemble snooker ball detector with rule-based consolidation.
    """

    EXPECTED_BALLS: Dict[str, int] = {
        "red": 15,
        "yellow": 1,
        "green": 1,
        "brown": 1,
        "blue": 1,
        "pink": 1,
        "black": 1,
        "white": 1,
    }

    COLOR_MAPPINGS: Dict[str, str] = {
        # Hyphenated
        "red-ball": "red",
        "yellow-ball": "yellow",
        "green-ball": "green",
        "brown-ball": "brown",
        "blue-ball": "blue",
        "pink-ball": "pink",
        "black-ball": "black",
        "white-ball": "white",
        "cue-ball": "white",
        # Capitalized
        "Red": "red",
        "Yellow": "yellow",
        "Green": "green",
        "Brown": "brown",
        "Blue": "blue",
        "Pink": "pink",
        "Black": "black",
        "White": "white",
        # Lowercase passthrough
        "red": "red",
        "yellow": "yellow",
        "green": "green",
        "brown": "brown",
        "blue": "blue",
        "pink": "pink",
        "black": "black",
        "white": "white",
    }

    def __init__(
        self,
        detectors: List[InferenceRunner],
        confidence_threshold: float = 0.3,
        temporal_window: int = 3,
        max_missing_frames: int = 3,
        missing_ball_confidence_factor: float = 0.6,
        **kwargs: Any,
    ) -> None:
        if not detectors:
            raise ValueError("At least one detector must be provided")

        self.detectors = detectors
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.max_missing_frames = max_missing_frames
        self.missing_ball_confidence_factor = missing_ball_confidence_factor

        # State
        self.previous_sanitized_results: Optional[List[Dict[str, Any]]] = None
        self.detection_history: deque[List[Dict[str, Any]]] = deque(maxlen=temporal_window)
        self.missing_ball_frames: Dict[str, int] = {}
        self.game_phase: str = "reds"
        self.expected_balls: Dict[str, int] = dict(self.EXPECTED_BALLS)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # --- Public API ---
    def predict(self, image: Union[str, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        # Step 1: run detectors and normalize
        all_predictions: List[Dict[str, Any]] = []
        detector_results: List[Dict[str, Any]] = []

        for i, detector in enumerate(self.detectors):
            try:
                result = detector.predict(image, **kwargs)
                detector_results.append(result)
                normalized = self._normalize_predictions(result.get("predictions", []))
                for p in normalized:
                    p["detector_index"] = i
                    if "detector_source" not in p or p["detector_source"] == "unknown":
                        p["detector_source"] = f"detector_{i+1}"
                all_predictions.extend(normalized)
                self.logger.info(f"Detector {i+1} found {len(normalized)} balls")
            except Exception as e:  # noqa: BLE001 - surface detector failures as warnings
                self.logger.warning(f"Detector {i+1} failed: {e}")
                detector_results.append({"predictions": []})

        # Step 2: size stats
        size_stats = self._compute_ball_size_stats(all_predictions)
        self.logger.debug(
            f"Ball size stats (px): median={size_stats['median']:.1f}, "
            f"p10={size_stats['p10']:.1f}, p90={size_stats['p90']:.1f}"
        )

        # Step 3: spatial grouping across detectors
        group_radius = 0.2 * size_stats["median"] if size_stats["median"] > 0 else 10.0
        groups = self._group_detections_across_detectors(all_predictions, group_radius)
        self.logger.info(
            f"Formed {len(groups)} spatial groups for consolidation (radius={group_radius:.1f}px)"
        )

        expected = self._expected_constraints()

        # Step 4: high certainty
        resolved_preds, remaining_groups = self._resolve_high_certainty(groups, size_stats, expected)
        if resolved_preds:
            self.logger.info(f"High-certainty resolved: {len(resolved_preds)} groups.")
        self.logger.info(f"Remaining {len(remaining_groups)} objects after high-certainty pass.")

        # Step 5: colored balls with confidence
        colored_preds, remaining_groups = self._resolve_colored_balls(
            remaining_groups, size_stats, expected, self.confidence_threshold
        )
        resolved_preds.extend(colored_preds)
        if colored_preds:
            self.logger.info(f"Resolved colored balls on confidence: {len(colored_preds)}.")
        self.logger.info(f"Remaining {len(remaining_groups)} objects after colored pass.")

        # Step 6: resolve remaining groups (mostly reds and ambiguous)
        remaining_preds = self._resolve_remaining(remaining_groups, size_stats, expected)
        resolved_preds.extend(remaining_preds)
        if remaining_preds:
            self.logger.info(f"Resolved remaining groups: {len(remaining_preds)}. {remaining_preds}")

        # Spatial sanity and temporal consistency
        resolved_preds = self._resolve_spatial_conflicts(resolved_preds)
        resolved_preds = self._apply_temporal_consistency(resolved_preds)

        # Update state
        self.previous_sanitized_results = resolved_preds
        self.detection_history.append(resolved_preds)

        # Image info passthrough
        image_info: Dict[str, Any] = {"width": 640, "height": 480}
        for res in detector_results:
            if "image" in res:
                image_info = res["image"]
                break

        return {
            "predictions": resolved_preds,
            "image": image_info,
            "model": "SnookerBallDetector",
            "sanitization_info": {
                "original_count": len(all_predictions),
                "final_count": len(resolved_preds),
                "ball_counts": self._count_balls_by_color(resolved_preds),
                "game_phase": self.game_phase,
                "size_stats": size_stats,
                "expected": expected,
            },
        }

    def predict_batch(self, images: List[Union[str, np.ndarray]], **kwargs: Any) -> List[Dict[str, Any]]:
        return [self.predict(img, **kwargs) for img in images]

    def visualize_predictions(
        self, image: Union[str, np.ndarray], predictions: Dict[str, Any], output_path: Optional[str] = None
    ) -> np.ndarray:
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()

        color_map = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "brown": (42, 42, 165),
            "blue": (255, 0, 0),
            "pink": (203, 192, 255),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
        }

        for pred in predictions.get("predictions", []):
            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])

            bcolor = pred["class"]
            color = color_map.get(bcolor, (128, 128, 128))

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{bcolor} {pred['confidence']:.2f}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if output_path:
            cv2.imwrite(output_path, img)

        return img

    def get_model_info(self) -> Dict[str, Any]:
        detector_info: List[str] = []
        for i, detector in enumerate(self.detectors):
            try:
                info = detector.get_model_info()
                detector_info.append(f"Detector {i+1}: {info.get('name', 'Unknown')}")
            except Exception:  # noqa: BLE001
                detector_info.append(f"Detector {i+1}: Info unavailable")
        return {
            "name": "SnookerBallDetector",
            "type": "ensemble_sanitizer",
            "detectors": detector_info,
            "confidence_threshold": self.confidence_threshold,
            "temporal_window": self.temporal_window,
            "max_missing_frames": self.max_missing_frames,
            "missing_ball_confidence_factor": self.missing_ball_confidence_factor,
        }

    # --- Core helpers ---
    def _normalize_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for pred in predictions:
            original_class = str(pred.get("class", ""))
            clazz = self.COLOR_MAPPINGS.get(original_class, original_class)
            if clazz not in self.EXPECTED_BALLS:
                continue
            normalized.append(
                {
                    "x": float(pred.get("x", 0.0)),
                    "y": float(pred.get("y", 0.0)),
                    "width": float(pred.get("width", 0.0)),
                    "height": float(pred.get("height", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                    "class": clazz,
                    "class_id": int(pred.get("class_id", 0)),
                    "detector_source": pred.get("detector_source", "unknown"),
                }
            )
        return normalized

    def _compute_ball_size_stats(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        if not predictions:
            return {"median": 0.0, "p10": 0.0, "p90": 0.0}
        diameters = [(p["width"] + p["height"]) / 2.0 for p in predictions]
        median = float(np.median(diameters))
        p10 = float(np.percentile(diameters, 10))
        p90 = float(np.percentile(diameters, 90))
        return {"median": median, "p10": p10, "p90": p90}

    def _group_detections_across_detectors(
        self, predictions: List[Dict[str, Any]], radius: float
    ) -> List[Dict[str, Any]]:
        if not predictions:
            return []
        groups: List[Dict[str, Any]] = []
        used = [False] * len(predictions)
        for i, p in enumerate(predictions):
            if used[i]:
                continue
            members = [p]
            used[i] = True
            cx, cy = p["x"], p["y"]
            for j in range(i + 1, len(predictions)):
                if used[j]:
                    continue
                q = predictions[j]
                if float(np.hypot(cx - q["x"], cy - q["y"])) <= radius:
                    members.append(q)
                    used[j] = True

            by_detector: Dict[int, Dict[str, Any]] = {}
            for m in members:
                idx = int(m.get("detector_index", -1))
                if idx not in by_detector or m["confidence"] > by_detector[idx]["confidence"]:
                    by_detector[idx] = m

            votes = self._vote_class_for_group(members)
            diam = float(np.median([(m["width"] + m["height"]) / 2.0 for m in members]))
            weight_sum = float(sum(m["confidence"] for m in members)) or 1.0
            wx = sum(m["x"] * m["confidence"] for m in members) / weight_sum
            wy = sum(m["y"] * m["confidence"] for m in members) / weight_sum
            groups.append(
                {
                    "members": members,
                    "by_detector": by_detector,
                    "centroid": (wx, wy),
                    "diameter": diam,
                    "votes": votes,
                }
            )
        return groups

    def _vote_class_for_group(self, members: List[Dict[str, Any]]) -> Dict[str, Any]:
        score: DefaultDict[str, float] = defaultdict(float)
        for m in members:
            score[m["class"]] += float(m["confidence"])
        if not score:
            return {"scores": {}, "top_class": None, "top_score": 0.0}
        top_class, top_score = max(score.items(), key=lambda kv: kv[1])
        return {"scores": dict(score), "top_class": top_class, "top_score": float(top_score)}

    def _aggregate_group_prediction(self, group: Dict[str, Any], clazz: str, target_size: float) -> Dict[str, Any]:
        x, y = group["centroid"]
        w = h = float(target_size)
        confs = [m["confidence"] for m in group["members"] if m["class"] == clazz] or [
            m["confidence"] for m in group["members"]
        ]
        base_conf = float(np.mean(confs))
        n_det = len(group["by_detector"])
        combined_conf = min(1.0, base_conf + 0.05 * max(0, n_det - 1))
        return {"x": float(x), "y": float(y), "width": w, "height": h, "confidence": combined_conf, "class": clazz, "class_id": 0}

    def _expected_constraints(self) -> Dict[str, Any]:
        expected_max = dict(self.expected_balls)
        prev = self.previous_sanitized_results or []
        prev_counts = self._count_balls_by_color(prev)
        red_low = prev_counts.get("red", 0)
        red_high = expected_max.get("red", 15)
        colored_low: Dict[str, int] = {}
        for c in ["yellow", "green", "brown", "blue", "pink", "black", "white"]:
            colored_low[c] = min(1, prev_counts.get(c, 0))
        return {
            "red_low": red_low,
            "red_high": red_high,
            "colored_low": colored_low,
            "colored_high": {c: 1 for c in colored_low.keys()},
        }

    def _resolve_high_certainty(
        self, groups: List[Dict[str, Any]], size_stats: Dict[str, float], expected: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        resolved: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        prev = self.previous_sanitized_results or []
        prev_by_color: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for p in prev:
            prev_by_color[p["class"]].append(p)

        for g in groups:
            n_det = len(g["by_detector"])
            size_ok = (size_stats["p10"] * 0.8 <= g["diameter"] <= size_stats["p90"] * 1.2) if size_stats["median"] > 0 else True
            top = g["votes"]["top_class"]
            has_prev_near = False
            if top is not None and prev_by_color.get(top):
                for pp in prev_by_color[top]:
                    if np.hypot(pp["x"] - g["centroid"][0], pp["y"] - g["centroid"][1]) <= 0.3 * max(1.0, size_stats["median"]):
                        has_prev_near = True
                        break
            if n_det >= 2 and size_ok and has_prev_near:
                pred = self._aggregate_group_prediction(g, top, size_stats["median"] or g["diameter"])
                pred["resolved_stage"] = "high_certainty"
                resolved.append(pred)
                self.logger.info(
                    f"High-certainty: kept {top} at ({pred['x']:.1f},{pred['y']:.1f}), "
                    f"detectors={n_det}, diam≈{g['diameter']:.1f}px"
                )
            else:
                remaining.append(g)

        return resolved, remaining

    def _resolve_colored_balls(
        self,
        groups: List[Dict[str, Any]],
        size_stats: Dict[str, float],
        expected: Dict[str, Any],
        conf_threshold: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        colors = ["blue", "black", "white", "green", "yellow", "brown", "pink"]
        resolved: List[Dict[str, Any]] = []
        leftover: List[Dict[str, Any]] = list(groups)

        assigned_colors = set()
        for color in colors:
            if expected["colored_low"].get(color, 0) >= 1 and color in assigned_colors:
                continue
            best_idx = -1
            best_score = -1.0
            for idx, g in enumerate(leftover):
                score = float(g["votes"]["scores"].get(color, 0.0))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0 and best_score > 0.0:
                g = leftover[best_idx]
                cand = self._aggregate_group_prediction(g, color, size_stats["median"] or g["diameter"])
                if cand["confidence"] >= conf_threshold:
                    cand["resolved_stage"] = "colored_confident"
                    resolved.append(cand)
                    assigned_colors.add(color)
                    leftover.pop(best_idx)
                    self.logger.info(
                        f"Colored: assigned {color} at ({cand['x']:.1f},{cand['y']:.1f}) "
                        f"conf={cand['confidence']:.2f}, votes={best_score:.2f}, detectors={len(g['by_detector'])}"
                    )
                else:
                    self.logger.info(
                        f"Colored: skipped {color} candidate due to low conf {cand['confidence']:.2f} < {conf_threshold:.2f}"
                    )

        return resolved, leftover

    def _resolve_remaining(
        self, groups: List[Dict[str, Any]], size_stats: Dict[str, float], expected: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        assigned_counts: DefaultDict[str, int] = defaultdict(int)

        def count_color(c: str) -> int:
            return assigned_counts[c] + sum(1 for p in output if p["class"] == c)

        # Ambiguous white/yellow swap if needed
        for g in list(groups):
            votes = g["votes"]["scores"]
            if "white" in votes and "yellow" in votes:
                if count_color("white") >= 1 and count_color("yellow") < expected["colored_high"]["yellow"]:
                    cand = self._aggregate_group_prediction(g, "yellow", size_stats["median"] or g["diameter"])
                    cand["confidence"] *= 0.8
                    cand["low_confidence_assignment"] = True
                    cand["resolved_stage"] = "ambiguous_white_yellow"
                    output.append(cand)
                    groups.remove(g)
                    self.logger.info(
                        f"Ambiguity: resolved as yellow at ({cand['x']:.1f},{cand['y']:.1f}) with reduced conf {cand['confidence']:.2f}"
                    )

        # Assign any remaining colored balls that are still missing
        for color in ["blue", "black", "white", "green", "yellow", "brown", "pink"]:
            if count_color(color) >= expected["colored_high"][color]:
                continue
            best_group = None
            best_score = 0.0
            for g in groups:
                sc = float(g["votes"]["scores"].get(color, 0.0))
                if sc > best_score:
                    best_score = sc
                    best_group = g
            if best_group is not None and best_score > 0.0:
                cand = self._aggregate_group_prediction(best_group, color, size_stats["median"] or best_group["diameter"])
                cand["resolved_stage"] = "colored_fill"
                output.append(cand)
                groups.remove(best_group)
                self.logger.info(
                    f"Fill: assigned {color} at ({cand['x']:.1f},{cand['y']:.1f}) conf={cand['confidence']:.2f}"
                )

        # Assign reds, including splitting oversized detections
        red_needed = max(0, expected["red_high"] - count_color("red"))
        if red_needed > 0:
            for g in list(groups):
                if red_needed <= 0:
                    break
                diam = g["diameter"] or size_stats["median"] or 0.0
                if size_stats["median"] > 0 and diam >= 1.7 * size_stats["median"] and red_needed >= 2:
                    # split into two reds for simplicity
                    pred_like = {
                        "x": g["centroid"][0],
                        "y": g["centroid"][1],
                        "width": diam,
                        "height": diam,
                        "confidence": min(1.0, (g["votes"]["top_score"] or 0.5) / max(1.0, len(g["members"]))),
                        "class": "red",
                        "class_id": 0,
                    }
                    splits = self._split_clustered_red_ball(pred_like, 2)
                    for s in splits:
                        s["resolved_stage"] = "red_split"
                    for s in splits:
                        if red_needed <= 0:
                            break
                        output.append(s)
                        red_needed -= 1
                    groups.remove(g)
                    self.logger.info(
                        f"Red: split oversized group at ({pred_like['x']:.1f},{pred_like['y']:.1f}) diam≈{diam:.1f}px"
                    )
                else:
                    cand = self._aggregate_group_prediction(g, "red", size_stats["median"] or diam)
                    cand["resolved_stage"] = "red_from_group"
                    output.append(cand)
                    red_needed -= 1
                    groups.remove(g)
                    self.logger.info(
                        f"Red: assigned from group at ({cand['x']:.1f},{cand['y']:.1f}) conf={cand['confidence']:.2f}"
                    )

        return output

    # --- Reused utilities ---
    def _split_clustered_red_ball(self, clustered_pred: Dict[str, Any], num_balls: int) -> List[Dict[str, Any]]:
        if num_balls <= 1:
            return [clustered_pred]
        split_balls: List[Dict[str, Any]] = []
        base_conf = float(clustered_pred.get("confidence", 0.5)) / num_balls
        if num_balls == 2:
            positions = [
                (clustered_pred["x"] - clustered_pred["width"] * 0.25, clustered_pred["y"]),
                (clustered_pred["x"] + clustered_pred["width"] * 0.25, clustered_pred["y"]),
            ]
        else:
            positions = [(clustered_pred["x"], clustered_pred["y"])] * num_balls
        individual_w = float(clustered_pred["width"]) / max(1.0, np.sqrt(num_balls))
        individual_h = float(clustered_pred["height"]) / max(1.0, np.sqrt(num_balls))
        for i, (x, y) in enumerate(positions):
            split_balls.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "width": individual_w,
                    "height": individual_h,
                    "confidence": base_conf,
                    "class": "red",
                    "class_id": 0,
                    "split_from_cluster": True,
                    "cluster_id": i,
                }
            )
        return split_balls

    def _resolve_spatial_conflicts(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not predictions:
            return predictions
        # simple NMS-like: if two colored balls overlap strongly, keep higher confidence
        colored = [p for p in predictions if p["class"] != "red"]
        reds = [p for p in predictions if p["class"] == "red"]

        kept: List[Dict[str, Any]] = []
        used = [False] * len(colored)
        for i, a in enumerate(colored):
            if used[i]:
                continue
            best = a
            used[i] = True
            ax1, ay1, ax2, ay2 = self._bbox(a)
            for j in range(i + 1, len(colored)):
                if used[j]:
                    continue
                b = colored[j]
                bx1, by1, bx2, by2 = self._bbox(b)
                if self._iou((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)) > 0.5:
                    # conflict: keep higher confidence of potentially different colors
                    if b["confidence"] > best["confidence"]:
                        self.logger.info(
                            f"Spatial conflict: replacing {best['class']} (conf={best['confidence']:.2f}) "
                            f"with {b['class']} (conf={b['confidence']:.2f}) due to overlap"
                        )
                        best = b
                    used[j] = True
            kept.append(best)

        kept.extend(reds)
        return kept

    def _apply_temporal_consistency(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.previous_sanitized_results:
            return predictions
        consistent = list(predictions)
        # Recover missing colored balls for a few frames
        prev = self.previous_sanitized_results
        current_colors = {p["class"] for p in predictions}
        recovered: List[Dict[str, Any]] = []
        for prev_pred in prev:
            color = prev_pred["class"]
            if color == "red":
                continue
            if color not in current_colors:
                missing = self.missing_ball_frames.get(color, 0)
                if missing < self.max_missing_frames:
                    rp = dict(prev_pred)
                    rp["confidence"] = float(rp["confidence"]) * self.missing_ball_confidence_factor
                    rp["from_previous_frame"] = True
                    rp["missing_frames"] = missing + 1
                    recovered.append(rp)
                    self.logger.info(
                        f"Temporal: recovered missing {color} (frames missing={missing + 1}) at "
                        f"({rp['x']:.1f},{rp['y']:.1f}) conf={rp['confidence']:.2f}"
                    )
        consistent.extend(recovered)

        # Update missing counters
        for prev_pred in prev:
            color = prev_pred["class"]
            if color == "red":
                continue
            if color in current_colors:
                self.missing_ball_frames.pop(color, None)
            else:
                self.missing_ball_frames[color] = self.missing_ball_frames.get(color, 0) + 1
        return consistent

    # --- small utilities --- 
    def _count_balls_by_color(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: DefaultDict[str, int] = defaultdict(int)
        for p in predictions:
            counts[p["class"]] += 1
        return dict(counts)

    def _bbox(self, p: Dict[str, Any]) -> Tuple[float, float, float, float]:
        x1 = p["x"] - p["width"] / 2.0
        y1 = p["y"] - p["height"] / 2.0
        x2 = p["x"] + p["width"] / 2.0
        y2 = p["y"] + p["height"] / 2.0
        return float(x1), float(y1), float(x2), float(y2)

    def _iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0


