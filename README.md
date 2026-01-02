# IdealMatch Practice Task - Video Object Detection

This repo contains multiple attempts to detect and label objects in videos, produce annotated outputs with bounding boxes and labels, and log detections to CSV/JSON. The final, best-performing setup is YOLOv8x-World v2 with optional tracking.

## Methodology (what I did)
1. Defined a construction-focused class list in `video_od/data/classes.json` to guide detection.
2. Tried several model families to balance accuracy, speed, and usability.
3. Produced annotated videos and CSV/JSON logs for each model.
4. Compared visual quality, label correctness, and runtime to select the best option.

## Models tried and how they were used
- YOLOv8x-World v2 (open-vocabulary detection, best overall)
  - Script: `video_od/construction_detect.py`
  - Uses `ultralytics` YOLO with open-vocabulary classes, configurable confidence/IoU.
  - Optional tracking via ByteTrack (track IDs on boxes).
  - Outputs: annotated video(s) + `*_detections.csv` + `*_detections.json`.

- YOLOv8 test sampler (sanity check)
  - Script: `video_od/yolo8_test.py`
  - Processes sparse frames from videos and exports annotated images.
  - Used to confirm labels and class list behavior quickly.

- MobileNetV3 SSDLite (COCO detector, lightweight baseline)
  - Script: `video_od/mobilenetv3_ssd_detect.py`
  - Direct detection, faster but less accurate for small/rare construction items.
  - Outputs annotated videos + CSV/JSON.

- ResNet18 / ResNet50 / ResNet101 (ImageNet classifiers + motion boxes)
  - Scripts: `video_od/resnet18_detect.py`, `video_od/resnet50_101_detect.py`
  - Uses background subtraction to propose motion boxes, then classifies each box.
  - Helpful as a fallback, but weaker localization and label accuracy.

- Video Swin (video classifier + motion boxes)
  - Script: `video_od/video_swin_detect.py`
  - Classifies short clips, applies labels to motion regions.
  - Worked, but less precise than YOLO for per-object boxes.

## Output artifacts
Generated outputs are organized only under `video_od/data/output/` (there is no `data/outputs` folder):
- Output videos are hosted on Google Drive due to GitHub size limits: https://drive.google.com/drive/folders/1Bcgtajncvl-Cy1sgUegaaEZ_VvgeVd6n?usp=sharing
- `yolov8x_worldv2/` for the final submission videos and logs.
- `mobilenetv3/`, `resnet18/`, `resnet50/`, `resnet101/`, `swin/` for baselines.
- `yolov8_test/` for sample annotated frames.

Each video run produces:
- Annotated video (`*_annotated.mp4` or model-specific name).
- `*_detections.csv` (frame/time/object/score/box).
- `*_detections.json` (structured per-frame detection log).

## Best-performing model
The best tradeoff between accuracy and runtime was **YOLOv8x-World v2**.
- It handled the custom construction class list better than the other options.
- It produced cleaner boxes, better label coverage, and stable detections.
- Tracking via ByteTrack improved interpretability across frames.

## Tools and libraries used
- Python, OpenCV, NumPy
- Ultralytics YOLO (YOLOv8-World v2)
- Torch/Torchvision (MobileNetV3-SSDLite, ResNet18/50/101, Video Swin)
- tqdm, PIL

## Challenges and limitations
- Non-YOLO backbones are classifiers rather than detectors, so I approximate localization with motion-based box proposals, which can be noisy.
- Open-vocabulary detection is sensitive to the exact class wording and scene context, so similar labels can yield different results.
- Small objects and low-light frames reduce visual signal, making detections less reliable across all models.

## Possible improvements
- Fine-tune a detector on construction-specific data.
- Add ReID-based tracking to reduce ID switches in crowded scenes.
- Calibrate confidence thresholds per class and add temporal smoothing.
- Use a stronger open-vocabulary model (e.g., grounded/segmenting detectors) if time allows.

## How to rerun (examples)
```powershell
# Best model with tracking
python video_od/construction_detect.py --input video_od/data --model yolov8x-worldv2.pt --tracker bytetrack.yaml

# Baselines
python video_od/mobilenetv3_ssd_detect.py --input video_od/data
python video_od/resnet18_detect.py --input video_od/data
python video_od/resnet50_101_detect.py --input video_od/data --model resnet101
python video_od/video_swin_detect.py --input video_od/data
```
