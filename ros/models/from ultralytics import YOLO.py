from ultralytics import YOLO
m = YOLO("yolov8n.pt")                 # 자동으로 pt 가중치 다운로드
m.export(format="onnx", opset=12, dynamic=True, simplify=True)
# 완료되면 yolov8n.onnx 가 생성됩니다 (보통 runs/weights/ 또는 현재 폴더).