# RobotUnified v2
- One-process (ROS + FastAPI) runner
- Robust YOLO box mapping (fix misaligned top-left boxes)
- Move endpoints: `/move/axis`, `/move/XYZ`, `/move/vision`
- Klipper driver via Moonraker

## Run
```bash
source /opt/ros/jazzy/setup.bash
pip install -r requirements.txt
cp /path/to/yolov5nu.onnx ros/models/
python3 main.py --host 0.0.0.0 --port 8000 \
  --device_index 0 --width 640 --height 480 --fps 15 \
  --onnx_path ros/models/yolov5nu.onnx --input_size 320 --conf_thres 0.25 \
  --onnx_format auto --debug_draw_letterbox \
  --use_klipper --klipper_config config/joints.yaml
```

## API JSON (Uppercase keys)
```json
{ "AXIS": "J1", "DIST": 100.0, "SPD": 0.5, "ACC": 1.0, "MODE": "relative" }
```
