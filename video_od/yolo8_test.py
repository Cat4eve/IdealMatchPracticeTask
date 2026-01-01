import argparse
import json
from pathlib import Path
from typing import List, Optional

import cv2
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

DEFAULT_CLASSES_FILE = "data/classes.json"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def is_image_path(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def is_video_path(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def expand_inputs(inputs: List[str], base_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if not p.exists() and not p.is_absolute():
            candidate = base_dir / p
            if candidate.exists():
                p = candidate
        if p.is_dir():
            for ext in sorted(IMAGE_EXTS | VIDEO_EXTS):
                paths.extend(sorted(p.glob(f"*{ext}")))
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


def annotate_and_save(result, out_path: Path) -> None:
    annotated = result.plot()
    cv2.imwrite(str(out_path), annotated)


def process_image(
    model: YOLO,
    image_path: Path,
    out_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: could not read image {image_path}")
        return
    results = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    out_path = out_dir / f"{image_path.stem}_yolo8.jpg"
    annotate_and_save(results[0], out_path)
    print(f"Saved: {out_path}")


def process_video(
    model: YOLO,
    video_path: Path,
    out_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    frame_step: int,
    max_frames: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_step > 1 and (frame_idx % frame_step) != 0:
                frame_idx += 1
                continue

            results = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
            out_path = out_dir / f"{video_path.stem}_frame{frame_idx:06d}_yolo8.jpg"
            annotate_and_save(results[0], out_path)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break
            frame_idx += 1
    finally:
        cap.release()

    print(f"Saved {saved} frames from {video_path.name}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quick YOLOv8 test using classes.json")
    ap.add_argument("--input", nargs="+", default=["data"],
                    help="Input image/video paths or a directory. Default: ./data")
    ap.add_argument("--out", default="data/output/yolov8_test",
                    help="Output directory for annotated images")
    ap.add_argument("--model", default="yolov8x-worldv2.pt",
                    help="YOLO model weights")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--device", default="0", help="Device: '0' for GPU 0, or 'cpu'")
    ap.add_argument("--classes", default=None,
                    help="Comma-separated class names (overrides classes file)")
    ap.add_argument("--classes-file", default=DEFAULT_CLASSES_FILE,
                    help="JSON file with class list.")
    ap.add_argument("--frame-step", type=int, default=30,
                    help="Process every Nth frame for videos")
    ap.add_argument("--max-frames", type=int, default=10,
                    help="Max frames to process per video (0 = no limit)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    ensure_dir(out_dir)

    inputs = expand_inputs(args.input, base_dir)
    if not inputs:
        raise RuntimeError("No inputs found from --input")

    model = YOLO(args.model)
    classes = parse_classes(args.classes) or load_classes_from_file(base_dir, args.classes_file)
    if classes and hasattr(model, "set_classes"):
        model.set_classes(classes)
        print(f"Using open-vocabulary classes: {', '.join(classes)}")
    elif classes:
        print("Warning: classes provided but model does not support open-vocabulary classes.")

    for p in inputs:
        if is_image_path(p):
            process_image(
                model=model,
                image_path=p,
                out_dir=out_dir,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
            )
        elif is_video_path(p):
            process_video(
                model=model,
                video_path=p,
                out_dir=out_dir,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                frame_step=max(1, args.frame_step),
                max_frames=max(0, args.max_frames),
            )
        else:
            print(f"Skipping unsupported file: {p}")


if __name__ == "__main__":
    main()
