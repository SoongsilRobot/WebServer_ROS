import cv2, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

class YoloNode(Node):
    """
    YOLO using Ultralytics (match your test.py behavior):
    - Loads model via ultralytics.YOLO(model_path)
    - Runs predict() directly on BGR frame
    - Publishes annotated image to /vision/streaming and detections to /vision/detections
    """
    def __init__(self, in_topic='/vision/streaming/raw', out_topic='/vision/streaming',
                 det_topic='/vision/detections', onnx_path='ros/models/yolov8n.onnx',
                 input_size=640, conf_thres=0.25, onnx_format='auto',  # kept for CLI compat
                 debug_draw_letterbox=False, preprocess='letterbox'):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        self.input_size = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = 0.45
        self.debug_draw_letterbox = bool(debug_draw_letterbox)

        qos_sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.pub_img = self.create_publisher(Image, out_topic, qos_sensor)
        self.pub_det = self.create_publisher(Detection2DArray, det_topic, qos_sensor)
        self.sub     = self.create_subscription(Image, in_topic, self.on_image, qos_sensor)

        self.model = None
        self.names = {}
        try:
            from ultralytics import YOLO
            self.model = YOLO(onnx_path)
            try:
                self.names = self.model.names or {}
            except Exception:
                self.names = {}
            self.get_logger().info(f'Ultralytics YOLO loaded: {onnx_path} (imgsz={self.input_size}, conf={self.conf_thres}, iou={self.iou_thres})')
        except Exception as e:
            self.model = None
            self.get_logger().error(f'Ultralytics model load failed: {e}')
            self.get_logger().warn('Continuing without inference (raw frames pass through).')

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        out, det_array = self.run_yolo(frame)
        det_array.header = msg.header
        out_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        self.pub_img.publish(out_msg)
        self.pub_det.publish(det_array)

    def run_yolo(self, frame):
        det_array = Detection2DArray()
        if self.model is None:
            return frame, det_array

        try:
            results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres,
                                         imgsz=self.input_size, verbose=False)
        except Exception as e:
            self.get_logger().warn(f'Inference failed: {e}')
            return frame, det_array

        H, W = frame.shape[0], frame.shape[1]
        annotated = frame.copy()

        boxes_all=[]; confs_all=[]; clses_all=[]
        try:
            for r in results:
                if getattr(r, 'boxes', None) is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy
                confs = r.boxes.conf
                clses = r.boxes.cls
                if hasattr(boxes, "cpu"): boxes = boxes.cpu().numpy()
                else: boxes = np.asarray(boxes)
                if hasattr(confs, "cpu"): confs = confs.cpu().numpy()
                else: confs = np.asarray(confs)
                if hasattr(clses, "cpu"): clses = clses.cpu().numpy().astype(int)
                else: clses = np.asarray(clses).astype(int)
                for b, c, k in zip(boxes, confs, clses):
                    boxes_all.append(b.tolist())
                    confs_all.append(float(c))
                    clses_all.append(int(k))
        except Exception as e:
            self.get_logger().warn(f'Result parsing failed: {e}')
            return frame, det_array

        for (x1,y1,x2,y2), c, cid in zip(boxes_all, confs_all, clses_all):
            label = self.names.get(int(cid), str(int(cid))) if isinstance(self.names, dict) else str(int(cid))
            cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(annotated, f"{label} {c:.2f}", (int(x1), max(0,int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

            det = Detection2D()
            det.bbox.center.position.x = float((x1 + x2) / 2.0)
            det.bbox.center.position.y = float((y1 + y2) / 2.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(cid))
            hyp.hypothesis.score = float(c)
            det.results.append(hyp)
            det_array.detections.append(det)

        return annotated, det_array