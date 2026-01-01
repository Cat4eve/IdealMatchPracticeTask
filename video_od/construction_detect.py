import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


DEFAULT_CLASSES_FILE = "data/classes.json"
DEFAULT_MODEL = "yolov8x-worldv2.pt"


@dataclass
class DetRow:
    frame_idx: int
    time_s: float
    track_id: Optional[int]
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slugify_model_name(model_path: str) -> str:
    stem = Path(model_path).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or "model"


def pick_video_writer(out_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")
    return writer


def draw_box(img: np.ndarray, xyxy: Tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_top = max(y1, th + 6)
    cv2.rectangle(img, (x1, y_top - th - 6), (x1 + tw + 6, y_top), (0, 255, 0), -1)
    cv2.putText(img, label, (x1 + 3, y_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def save_csv(rows: List[DetRow], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "track_id", "cls_id", "cls_name", "conf", "x1", "y1", "x2", "y2"])
        for r in rows:
            w.writerow([r.frame_idx, f"{r.time_s:.6f}", r.track_id, r.cls_id, r.cls_name, f"{r.conf:.6f}",
                        f"{r.x1:.2f}", f"{r.y1:.2f}", f"{r.x2:.2f}", f"{r.y2:.2f}"])


def save_json(rows: List[DetRow], out_json: Path) -> None:
    by_frame: Dict[int, Dict] = {}
    for r in rows:
        if r.frame_idx not in by_frame:
            by_frame[r.frame_idx] = {"frame": r.frame_idx, "time_s": r.time_s, "objects": []}
        by_frame[r.frame_idx]["objects"].append({
            "track_id": r.track_id,
            "class_id": r.cls_id,
            "class_name": r.cls_name,
            "confidence": r.conf,
            "bbox_xyxy": [r.x1, r.y1, r.x2, r.y2],
        })
    payload = [by_frame[k] for k in sorted(by_frame.keys())]
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_classes(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    classes = [c.strip() for c in raw.split(",") if c.strip()]
    return classes or None


def load_classes_from_file(base_dir: Path, classes_file: str) -> List[str]:
    classes_path = Path(classes_file)
    if not classes_path.is_absolute():
        classes_path = (base_dir / classes_path).resolve()
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    payload = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"Classes file must be a JSON array of strings: {classes_path}")
    classes = [item.strip() for item in payload if item.strip()]
    if not classes:
        raise ValueError(f"Classes file is empty: {classes_path}")
    return classes


def expand_inputs(inputs: List[str], base_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if not p.exists() and not p.is_absolute():
            candidate = base_dir / p
            if candidate.exists():
                p = candidate
        if p.is_dir():
            for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v", "*.webm"):
                paths.extend(sorted(p.glob(ext)))
        else:
            paths.append(p)
    seen = set()
    out = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def process_video(
    model: YOLO,
    video_path: Path,
    out_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    tracker: Optional[str],
    show_track_ids: bool,
    max_frames: int,
    save_csv_flag: bool,
    save_json_flag: bool,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    stem = video_path.stem
    out_video = out_dir / f"{stem}_annotated.mp4"
    out_csv = out_dir / f"{stem}_detections.csv"
    out_json = out_dir / f"{stem}_detections.json"

    writer = pick_video_writer(out_video, fps, width, height)
    all_rows: List[DetRow] = []
    frame_idx = 0

    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc=f"Processing {video_path.name}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            time_s = frame_idx / fps

            if tracker:
                results_list = model.track(
                    frame,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                    verbose=False,
                    persist=True,
                    tracker=tracker,
                )
            else:
                results_list = model.predict(
                    frame,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                    verbose=False,
                )

            res = results_list[0]
            names = res.names

            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)

                ids = None
                if hasattr(res.boxes, "id") and res.boxes.id is not None:
                    ids = res.boxes.id.cpu().numpy().astype(int)

                for j in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[j].tolist()
                    cls_id = int(clss[j])
                    cls_name = names.get(cls_id, str(cls_id))
                    c = float(confs[j])
                    track_id = int(ids[j]) if ids is not None else None

                    xi1 = int(max(0, min(width - 1, round(x1))))
                    yi1 = int(max(0, min(height - 1, round(y1))))
                    xi2 = int(max(0, min(width - 1, round(x2))))
                    yi2 = int(max(0, min(height - 1, round(y2))))

                    if show_track_ids and track_id is not None:
                        label = f"ID:{track_id} {cls_name} {c:.2f}"
                    else:
                        label = f"{cls_name} {c:.2f}"

                    draw_box(frame, (xi1, yi1, xi2, yi2), label)

                    all_rows.append(DetRow(
                        frame_idx=frame_idx,
                        time_s=time_s,
                        track_id=track_id,
                        cls_id=cls_id,
                        cls_name=cls_name,
                        conf=c,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ))

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

            if max_frames > 0 and frame_idx >= max_frames:
                break
    finally:
        pbar.close()
        cap.release()
        writer.release()

    if save_csv_flag:
        save_csv(all_rows, out_csv)
    if save_json_flag:
        save_json(all_rows, out_json)

    print(f"\nSaved: {out_video}")
    if save_csv_flag:
        print(f"Saved: {out_csv}")
    if save_json_flag:
        print(f"Saved: {out_json}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Construction-focused video detection using YOLOv8-World + ByteTrack"
    )
    ap.add_argument("--input", nargs="+", default=["data"],
                    help="Input video path(s) or a directory. Default: ./data")
    ap.add_argument("--out", default=None,
                    help="Output directory. Default: ./data/output/<model_name>")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="YOLO model weights. Recommended: yolov8x-worldv2.pt")
    ap.add_argument("--classes", default=None,
                    help="Comma-separated class names for open-vocabulary models.")
    ap.add_argument("--classes-file", default=DEFAULT_CLASSES_FILE,
                    help="JSON file with class list for open-vocabulary models.")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--device", default="0",
                    help="Device: '0' for GPU 0, 'cpu' for CPU")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="Tracker yaml path. Set to '' to disable tracking.")
    ap.add_argument("--no-track-ids", action="store_true", help="Do not draw track IDs on the video")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Limit frames per video (0 = process all frames)")
    ap.add_argument("--no-csv", action="store_true", help="Do not write CSV detections")
    ap.add_argument("--no-json", action="store_true", help="Do not write JSON detections")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    model_tag = slugify_model_name(args.model)
    if args.out is None:
        out_dir = (base_dir / "data" / "output" / model_tag).resolve()
    else:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = (base_dir / out_dir).resolve()
    ensure_dir(out_dir)

    video_paths = expand_inputs(args.input, base_dir)
    if not video_paths:
        raise RuntimeError("No videos found from --input")

    tracker = args.tracker.strip()
    if tracker == "":
        tracker = None
    else:
        tracker_path = Path(tracker)
        if not tracker_path.exists():
            candidate = base_dir / tracker
            if candidate.exists():
                tracker_path = candidate
            else:
                raise FileNotFoundError(f"Tracker config not found: {tracker_path.resolve()}")
        tracker = str(tracker_path.resolve())

    model = YOLO(args.model)
    model_name = Path(args.model).name.lower()
    classes = parse_classes(args.classes)
    if classes is None and "world" in model_name:
        classes = load_classes_from_file(base_dir, args.classes_file)
    if classes:
        if hasattr(model, "set_classes"):
            model.set_classes(classes)
            print(f"Using open-vocabulary classes: {', '.join(classes)}")
        else:
            print("Warning: --classes ignored (model does not support open-vocabulary classes).")

    for vp in video_paths:
        process_video(
            model=model,
            video_path=vp,
            out_dir=out_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            tracker=tracker,
            show_track_ids=not args.no_track_ids,
            max_frames=max(0, args.max_frames),
            save_csv_flag=not args.no_csv,
            save_json_flag=not args.no_json,
        )


if __name__ == "__main__":
    main()
