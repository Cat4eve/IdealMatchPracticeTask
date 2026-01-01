import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
import torchvision.models.video as tv_video


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
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 255), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_top = max(y1, th + 6)
    cv2.rectangle(img, (x1, y_top - th - 6), (x1 + tw + 6, y_top), (0, 180, 255), -1)
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


def load_swin_model(name: str):
    name = name.lower()
    if name in {"swin_t", "swin3d_t"} and hasattr(tv_video, "Swin3D_T_Weights"):
        weights = tv_video.Swin3D_T_Weights.DEFAULT
        model = tv_video.swin3d_t(weights=weights)
        return model, weights
    if name in {"swin_s", "swin3d_s"} and hasattr(tv_video, "Swin3D_S_Weights"):
        weights = tv_video.Swin3D_S_Weights.DEFAULT
        model = tv_video.swin3d_s(weights=weights)
        return model, weights
    if name in {"swin_b", "swin3d_b"} and hasattr(tv_video, "Swin3D_B_Weights"):
        weights = tv_video.Swin3D_B_Weights.DEFAULT
        model = tv_video.swin3d_b(weights=weights)
        return model, weights
    raise RuntimeError("Video Swin model is not available in this torchvision build.")


def match_label_to_class(label: str, allowed: List[str]) -> Optional[str]:
    label_l = label.lower()
    for cls in allowed:
        cls_l = cls.lower()
        if " " in cls_l or "-" in cls_l:
            if cls_l in label_l:
                return cls
        else:
            if re.search(rf"\b{re.escape(cls_l)}\b", label_l):
                return cls
    return None


def compute_motion_boxes(
    masks: List[np.ndarray],
    min_area: int,
    max_boxes: int,
) -> List[Tuple[int, int, int, int]]:
    if not masks:
        return []
    union = masks[0].copy()
    for m in masks[1:]:
        union = cv2.bitwise_or(union, m)
    union = cv2.medianBlur(union, 5)
    _, union = cv2.threshold(union, 200, 255, cv2.THRESH_BINARY)
    union = cv2.morphologyEx(union, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append((x, y, x + w, y + h, w * h))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in boxes[:max_boxes]]


def process_video(
    model,
    preprocess,
    categories: List[str],
    allowed_classes: List[str],
    video_path: Path,
    out_dir: Path,
    conf: float,
    device: torch.device,
    clip_len: int,
    topk: int,
    min_area: int,
    max_boxes: int,
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
    out_video = out_dir / f"{stem}_swin.mp4"
    out_csv = out_dir / f"{stem}_detections.csv"
    out_json = out_dir / f"{stem}_detections.json"

    writer = pick_video_writer(out_video, fps, width, height)
    all_rows: List[DetRow] = []
    frame_idx = 0
    clip_frames: List[np.ndarray] = []
    clip_times: List[float] = []
    clip_masks: List[np.ndarray] = []
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc=f"Processing {video_path.name}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            clip_frames.append(frame)
            clip_times.append(frame_idx / fps)
            clip_masks.append(fgbg.apply(frame))
            frame_idx += 1
            pbar.update(1)

            if len(clip_frames) < clip_len:
                continue

            boxes = compute_motion_boxes(clip_masks, min_area=min_area, max_boxes=max_boxes)
            if not boxes:
                boxes = [(0, 0, width - 1, height - 1)]

            for (x1, y1, x2, y2) in boxes:
                cropped = [f[y1:y2, x1:x2] for f in clip_frames]
                label, score = classify_clip(
                    model, preprocess, categories, allowed_classes, cropped, device, topk
                )
                if not label or score < conf:
                    continue
                for f, t in zip(clip_frames, clip_times):
                    draw_box(f, (x1, y1, x2, y2), f"{label} {score:.2f}")
                    all_rows.append(DetRow(
                        frame_idx=int(round(t * fps)),
                        time_s=t,
                        cls_name=label,
                        conf=score,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ))

            for f in clip_frames:
                writer.write(f)

            clip_frames = []
            clip_times = []
            clip_masks = []

        if clip_frames:
            boxes = compute_motion_boxes(clip_masks, min_area=min_area, max_boxes=max_boxes)
            if not boxes:
                boxes = [(0, 0, width - 1, height - 1)]
            for (x1, y1, x2, y2) in boxes:
                cropped = [f[y1:y2, x1:x2] for f in clip_frames]
                label, score = classify_clip(
                    model, preprocess, categories, allowed_classes, cropped, device, topk
                )
                if not label or score < conf:
                    continue
                for f, t in zip(clip_frames, clip_times):
                    draw_box(f, (x1, y1, x2, y2), f"{label} {score:.2f}")
                    all_rows.append(DetRow(
                        frame_idx=int(round(t * fps)),
                        time_s=t,
                        cls_name=label,
                        conf=score,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ))
            for f in clip_frames:
                writer.write(f)
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


def classify_clip(
    model,
    preprocess,
    categories: List[str],
    allowed_classes: List[str],
    clip_frames: List[np.ndarray],
    device: torch.device,
    topk: int,
) -> Tuple[Optional[str], float]:
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip_frames]
    clip = torch.from_numpy(np.stack(rgb_frames)).permute(0, 3, 1, 2)  # T,C,H,W
    clip = preprocess(clip).to(device)
    with torch.inference_mode():
        logits = model(clip.unsqueeze(0))[0]
    probs = torch.softmax(logits, dim=0)
    top_probs, top_idxs = torch.topk(probs, k=topk)

    for prob, idx in zip(top_probs, top_idxs):
        label = categories[int(idx)]
        matched = match_label_to_class(label, allowed_classes)
        if matched is not None:
            return matched, float(prob)
    return None, 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Video Swin Transformer classification on videos")
    ap.add_argument("--input", nargs="+", default=["data"],
                    help="Input video path(s) or a directory. Default: ./data")
    ap.add_argument("--out", default="data/output/swin",
                    help="Output directory. Default: ./data/output/swin")
    ap.add_argument("--model", default="swin_t",
                    help="Model variant: swin_t, swin_s, swin_b")
    ap.add_argument("--device", default="0", help="Device: '0' for GPU 0, or 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--clip-len", type=int, default=32, help="Clip length in frames")
    ap.add_argument("--topk", type=int, default=5, help="Top-K labels to check for a match")
    ap.add_argument("--min-area", type=int, default=900, help="Minimum motion box area")
    ap.add_argument("--max-boxes", type=int, default=6, help="Maximum boxes per clip")
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
    model, weights = load_swin_model(args.model)
    model = model.to(device)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta.get("categories", [])
    if not categories:
        raise RuntimeError("Model categories not found in weights metadata.")

    classes = parse_classes(args.classes) or load_classes_from_file(base_dir, args.classes_file)

    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print("Note: Video Swin is a classifier. This script uses motion boxes per clip.")

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
            clip_len=max(1, args.clip_len),
            topk=max(1, args.topk),
            min_area=max(1, args.min_area),
            max_boxes=max(1, args.max_boxes),
            save_csv_flag=not args.no_csv,
            save_json_flag=not args.no_json,
        )


if __name__ == "__main__":
    main()
