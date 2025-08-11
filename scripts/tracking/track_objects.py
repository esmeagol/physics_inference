#!/usr/bin/env python3
"""
Track snooker balls in a video using SnookerBallDetector + ByteTrack.

- Uses LocalPT as the underlying detector within SnookerBallDetector
- Tracks with supervision.ByteTrack
- Writes annotated video
"""

from __future__ import annotations

import os
import argparse
import time
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.detection.local_pt_inference import LocalPT
from src.detection.snooker_ball_detector import SnookerBallDetector


COLOR_TO_ID: Dict[str, int] = {
    "red": 1,
    "yellow": 2,
    "green": 3,
    "brown": 4,
    "blue": 5,
    "pink": 6,
    "black": 7,
    "white": 8,
}
ID_TO_COLOR = {v: k for k, v in COLOR_TO_ID.items()}


def predictions_to_sv(preds: List[Dict], frame_hw: Tuple[int, int]) -> sv.Detections:
    if not preds:
        return sv.Detections.empty()
    boxes = []
    confidences = []
    class_ids = []
    h, w = frame_hw
    for p in preds:
        cx, cy, pw, ph = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"]) 
        x1 = max(0.0, cx - pw / 2.0)
        y1 = max(0.0, cy - ph / 2.0)
        x2 = min(float(w), cx + pw / 2.0)
        y2 = min(float(h), cy + ph / 2.0)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(p.get("confidence", 0.0)))
        class_ids.append(int(COLOR_TO_ID.get(str(p.get("class", "")).lower(), 0)))
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int) if any(c != 0 for c in class_ids) else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track snooker balls using SnookerBallDetector + ByteTrack"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/abhinavrai/Playground/snooker_data/trained models/ar-snkr_objd-lolhi-3-yolov11-medium-weights.pt",
        help="Path to model weights (.pt file)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/test_videos/filtered_ROS-Frame-2.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/output/tracked_video.mp4",
        help="Path to save output video",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold (0-1)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU threshold for NMS (0-1)"
    )
    parser.add_argument(
        "--fps-limit", type=float, default=None, help="Limit processing FPS"
    )
    parser.add_argument(
        "--start-time", type=float, default=0, help="Start time in seconds"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Duration to process in seconds"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show preview window during processing"
    )
    parser.add_argument(
        "--no-traces", action="store_true", help="Disable drawing motion traces"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["bytetrack", "deepsort"],
        default="bytetrack",
        help="Tracker to use",
    )
    # ByteTrack tuning
    parser.add_argument("--bt-activation", type=float, default=0.25, help="ByteTrack track_activation_threshold")
    parser.add_argument("--bt-lost", type=int, default=60, help="ByteTrack lost_track_buffer")
    parser.add_argument("--bt-min-match", type=float, default=0.75, help="ByteTrack minimum_matching_threshold")
    # DeepSort tuning
    parser.add_argument("--ds-max-age", type=int, default=60, help="DeepSort max_age (frames)")
    parser.add_argument("--ds-n-init", type=int, default=3, help="DeepSort n_init (consecutive hits to confirm)")
    parser.add_argument("--ds-max-iou-distance", type=float, default=0.7, help="DeepSort max_iou_distance")

    args = parser.parse_args()

    # Verify input files exist
    if not os.path.isfile(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return

    if not os.path.isfile(args.video):
        print(f"Error: Video file does not exist: {args.video}")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build SnookerBallDetector with a LocalPT base detector
    print(f"Initializing SnookerBallDetector with model: {args.model}")
    base_detector = LocalPT(
        model_path=args.model, confidence=args.confidence, iou=args.iou
    )
    sbd = SnookerBallDetector(
        detectors=[base_detector],
        confidence_threshold=max(0.3, args.confidence * 0.8),
    )

    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    trace_annotator = sv.TraceAnnotator(thickness=2)

    # Process video
    print(f"Processing video: {args.video}")
    print(f"Output will be saved to: {args.output}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start_time * fps)
    end_frame = (
        start_frame + int(args.duration * fps) if args.duration else total_frames
    )

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    fps_delay = 1.0 / args.fps_limit if args.fps_limit else 0.0

    # Initialize chosen tracker
    tracker_kind = args.tracker.lower()
    bytetrack = None
    ds_tracker = None
    if tracker_kind == "bytetrack":
        bytetrack = sv.ByteTrack(
            track_activation_threshold=float(args.bt_activation),
            lost_track_buffer=int(args.bt_lost),
            minimum_matching_threshold=float(args.bt_min_match),
            frame_rate=int(round(fps)) if fps and fps > 0 else 30,
        )
        print(
            f"Using ByteTrack: activation={args.bt_activation}, lost={args.bt_lost}, min_match={args.bt_min_match}, fps={int(round(fps)) if fps else 30}"
        )
    else:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except Exception as e:
            print(
                "Error: deep-sort-realtime not installed. Install with: pip install deep-sort-realtime\n"
                f"Import error: {e}"
            )
            return
        ds_tracker = DeepSort(
            max_age=int(args.ds_max_age),
            n_init=int(args.ds_n_init),
            max_iou_distance=float(args.ds_max_iou_distance),
        )
        print(
            f"Using DeepSort (deep-sort-realtime): max_age={args.ds_max_age}, n_init={args.ds_n_init}, max_iou_distance={args.ds_max_iou_distance}"
        )

    start_time = time.time()
    frame_idx = start_frame
    try:
        while frame_idx < end_frame:
            loop_t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Detect with SnookerBallDetector
            det = sbd.predict(frame, confidence=args.confidence)
            preds = det.get("predictions", [])

            annotated = frame.copy()

            if tracker_kind == "bytetrack":
                # Convert to supervision detections and update ByteTrack
                sv_dets = predictions_to_sv(preds, (height, width))
                sv_dets = bytetrack.update_with_detections(sv_dets)

                # Build labels
                labels: List[str] = []
                for i in range(len(sv_dets)):
                    tid = int(sv_dets.tracker_id[i]) if sv_dets.tracker_id is not None else i
                    cls_id = int(sv_dets.class_id[i]) if sv_dets.class_id is not None else 0
                    cls_name = ID_TO_COLOR.get(cls_id, "ball")
                    labels.append(f"{cls_name} #{tid}")

                # Annotate
                if not args.no_traces and len(sv_dets) > 0:
                    annotated = trace_annotator.annotate(scene=annotated, detections=sv_dets)
                if len(sv_dets) > 0:
                    annotated = box_annotator.annotate(scene=annotated, detections=sv_dets)
                    # Draw labels manually for compatibility
                    for i, label in enumerate(labels):
                        if i < len(sv_dets):
                            x1, y1, x2, y2 = sv_dets.xyxy[i]
                            cv2.putText(
                                annotated,
                                label,
                                (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )
            else:
                # DeepSort expects detections as ([left, top, width, height], conf, class_id)
                detections = []
                for p in preds:
                    cx, cy, pw, ph = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"]) 
                    left = max(0.0, cx - pw / 2.0)
                    top = max(0.0, cy - ph / 2.0)
                    conf = float(p.get("confidence", 0.0))
                    cls_id = int(COLOR_TO_ID.get(str(p.get("class", "")).lower(), 0))
                    detections.append(([left, top, pw, ph], conf, cls_id))

                try:
                    tracks = ds_tracker.update_tracks(detections, frame=frame)
                except Exception as e:
                    print(f"DeepSort update_tracks error: {e}")
                    tracks = []

                # Convert tracks to sv.Detections for annotation
                xyxy_list: List[List[float]] = []
                tracker_ids: List[int] = []
                class_ids: List[int] = []
                ds_labels: List[str] = []
                for trk in tracks:
                    try:
                        if hasattr(trk, "is_confirmed") and not trk.is_confirmed():
                            continue
                        if getattr(trk, "time_since_update", 0) > 0:
                            continue
                        l, t, r, b = trk.to_ltrb()  # left, top, right, bottom
                        xyxy_list.append([float(l), float(t), float(r), float(b)])
                        tid = int(trk.track_id)
                        tracker_ids.append(tid)
                        # Try to get class id/name if available
                        cls_id = 0
                        cls_name = "ball"
                        if hasattr(trk, "get_det_class"):
                            try:
                                det_cls = trk.get_det_class()
                                if det_cls is not None:
                                    cls_id = int(det_cls)
                                    cls_name = ID_TO_COLOR.get(cls_id, "ball")
                            except Exception:
                                pass
                        class_ids.append(cls_id)
                        ds_labels.append(f"{cls_name} #{tid}")
                    except Exception:
                        continue

                if xyxy_list:
                    dets = sv.Detections(
                        xyxy=np.array(xyxy_list, dtype=np.float32),
                        class_id=np.array(class_ids, dtype=int) if any(class_ids) else None,
                        tracker_id=np.array(tracker_ids, dtype=int),
                    )
                    if not args.no_traces:
                        annotated = trace_annotator.annotate(scene=annotated, detections=dets)
                    annotated = box_annotator.annotate(scene=annotated, detections=dets)
                    for i, label in enumerate(ds_labels):
                        if i < len(dets):
                            x1, y1, x2, y2 = dets.xyxy[i]
                            cv2.putText(
                                annotated,
                                label,
                                (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )

            out.write(annotated)
            if args.preview:
                cv2.imshow("Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

            # Limit FPS if specified
            if fps_delay:
                elapsed = time.time() - loop_t0
                sleep_time = max(0.0, fps_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    finally:
        cap.release()
        out.release()
        if args.preview:
            cv2.destroyAllWindows()

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
