# YOLOv8n-BCD
Code for "An Efficient and Lightweight Model for Traffic Object Detection in Autonomous Vehicles under Harsh Nighttime Conditions"


## Files

- `yolov8n-BiFPN+CA+Dy_Sample.yaml` – model configuration
- `BiFPN.py` – Bidirectional Feature Pyramid Network
- `CoordAtt.py` – Coordinate Attention module
- `Dvsample.py` – DySample dynamic upsampler
- `mytrain.py` – training script

## Requirements

Install dependencies:

```bash
pip install ultralytics torch numpy opencv-python


## Dataset

The BDD100K dataset is publicly available at [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/).  
We use a custom nighttime subset (3,500 images) derived from it. Please refer to the paper for the exact filtering and splitting criteria.
