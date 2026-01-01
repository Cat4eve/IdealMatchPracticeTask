import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class DetRow:
    frame_idx: int
    time_s: float
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


DEFAULT_CLASSES_FILE = "data/classes.json"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_video_writer(out_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")
    return writer


def draw_box(img: np.ndarray, xyxy: Tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 255), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_top = max(y1, th + 6)
    cv2.rectangle(img, (x1, y_top - th - 6), (x1 + tw + 6, y_top), (80, 200, 255), -1)
    cv2.putText(img, label, (x1 + 3, y_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def save_csv(rows: List[DetRow], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "cls_name", "conf", "x1", "y1", "x2", "y2"])
        for r in rows:
            w.writerow([r.frame_idx, f"{r.time_s:.6f}", r.cls_name, f"{r.conf:.6f}",
                        f"{r.x1:.2f}", f"{r.y1:.2f}", f"{r.x2:.2f}", f"{r.y2:.2f}"])


def save_json(rows: List[DetRow], out_json: Path) -> None:
    by_frame: Dict[int, Dict] = {}
    for r in rows:
        if r.frame_idx not in by_frame:
            by_frame[r.frame_idx] = {"frame": r.frame_idx, "time_s": r.time_s, "objects": []}
        by_frame[r.frame_idx]["objects"].append({
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


def select_device(device_str: str) -> torch.device:
    d = device_str.strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{d}")
        raise ValueError("CUDA not available; use --device cpu")
    if d.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(d)
        raise ValueError("CUDA not available; use --device cpu")
    return torch.device("cpu")


def map_to_allowed(label: str, allowed: List[str]) -> Optional[str]:
    label_l = label.lower()
    for cls in allowed:
        cls_l = cls.lower()
        if label_l == cls_l:
            return cls
    for cls in allowed:
        cls_l = cls.lower()
        if cls_l in label_l or label_l in cls_l:
            return cls
    return None


def process_video(
    model,
    preprocess,
    categories: List[str],
    allowed_classes: List[str],
    video_path: Path,
    out_dir: Path,
    conf: float,
    device: torch.device,
    min_area: int,
    max_boxes: int,
    topk: int,
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
    out_video = out_dir / f"{stem}_resnet18.mp4"
    out_csv = out_dir / f"{stem}_detections.csv"
    out_json = out_dir / f"{stem}_detections.json"

    writer = pick_video_writer(out_video, fps, width, height)
    all_rows: List[DetRow] = []
    frame_idx = 0

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc=f"Processing {video_path.name}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            time_s = frame_idx / fps
            mask = fgbg.apply(frame)
            mask = cv2.medianBlur(mask, 5)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < min_area:
                    continue
                boxes.append((x, y, x + w, y + h, w * h))
            boxes.sort(key=lambda b: b[4], reverse=True)
            boxes = boxes[:max_boxes]

            for (x1, y1, x2, y2, _) in boxes:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                tensor = preprocess(pil).to(device)
                with torch.inference_mode():
                    logits = model(tensor.unsqueeze(0))[0]
                probs = torch.softmax(logits, dim=0)
                top_probs, top_idxs = torch.topk(probs, k=topk)

                matched = None
                matched_conf = 0.0
                for prob, idx in zip(top_probs, top_idxs):
                    label = categories[int(idx)]
                    mapped = map_to_allowed(label, allowed_classes)
                    if mapped is not None:
                        matched = mapped
                        matched_conf = float(prob)
                        break
                if matched is None or matched_conf < conf:
                    continue

                label_text = f"{matched} {matched_conf:.2f}"
                draw_box(frame, (x1, y1, x2, y2), label_text)
                all_rows.append(DetRow(
                    frame_idx=frame_idx,
                    time_s=time_s,
                    cls_name=matched,
                    conf=matched_conf,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                ))

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)
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
    ap = argparse.ArgumentParser(description="ResNet-18 + background subtraction video detection")
    ap.add_argument("--input", nargs="+", default=["data"],
                    help="Input video path(s) or a directory. Default: ./data")
    ap.add_argument("--out", default="data/output/resnet18",
                    help="Output directory. Default: ./data/output/resnet18")
    ap.add_argument("--device", default="0", help="Device: '0' for GPU 0, or 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--min-area", type=int, default=900,
                    help="Minimum motion box area")
    ap.add_argument("--max-boxes", type=int, default=8,
                    help="Maximum boxes per frame")
    ap.add_argument("--topk", type=int, default=5,
                    help="Top-K ImageNet labels to check for a match")
    ap.add_argument("--classes", default=None,
                    help="Comma-separated class names (overrides classes file)")
    ap.add_argument("--classes-file", default=DEFAULT_CLASSES_FILE,
                    help="JSON file with class list.")
    ap.add_argument("--no-csv", action="store_true", help="Do not write CSV detections")
    ap.add_argument("--no-json", action="store_true", help="Do not write JSON detections")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    ensure_dir(out_dir)

    video_paths = expand_inputs(args.input, base_dir)
    if not video_paths:
        raise RuntimeError("No videos found from --input")

    device = select_device(args.device)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    classes = parse_classes(args.classes) or load_classes_from_file(base_dir, args.classes_file)

    print(f"Using device: {device}")
    print("Note: ResNet-18 is a classifier. This script uses motion boxes + classification.")

    for vp in video_paths:
        process_video(
            model=model,
            preprocess=preprocess,
            categories=categories,
            allowed_classes=classes,
            video_path=vp,
            out_dir=out_dir,
            conf=args.conf,
            device=device,
            min_area=args.min_area,
            max_boxes=args.max_boxes,
            topk=max(1, args.topk),
            save_csv_flag=not args.no_csv,
            save_json_flag=not args.no_json,
        )


if __name__ == "__main__":
    main()
