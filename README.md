# YOLOv8n-BCD
Code for "An Efficient and Lightweight Model for Traffic Object Detection in Autonomous Vehicles under Nighttime Conditions"


## Files

- `yolov8n-BiFPN+CA+Dy_Sample.yaml` – model configuration
- `BiFPN.py` – Bidirectional Feature Pyramid Network module
- `CoordAtt.py` – Coordinate Attention module
- `Dvsample.py` – DySample dynamic upsampler
- `mytrain.py` – training script

## Dataset

The BDD100K dataset is publicly available at [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/).  


## Requirements

Install dependencies:

```bash
pip install ultralytics torch numpy opencv-python


