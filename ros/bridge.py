import threading
from typing import Optional, List
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

class ROSBridge(Node):
    def __init__(self):
        super().__init__('ros_bridge')
        self.bridge = CvBridge()
        self._jpeg: Optional[bytes] = None
        self._det = {}
        self._lock = threading.Lock()

        qos_sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, '/vision/streaming', self.on_image, qos_sensor)
        self.create_subscription(Detection2DArray, '/vision/detections', self.on_det, qos_sensor)

        self.pub_movej = self.create_publisher(Float64MultiArray, '/robot/movej_cmd', 10)
        self.pub_move_axis = self.create_publisher(Float64MultiArray, '/robot/move_axis_cmd', 10)
        self.pub_move_xyz  = self.create_publisher(Float64MultiArray, '/robot/move_xyz_cmd', 10)
        self.pub_move_vision = self.create_publisher(Float64MultiArray, '/robot/move_vision_cmd', 10)

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with self._lock: self._jpeg = jpg.tobytes()

    def on_det(self, msg: Detection2DArray):
        dets = []
        for d in msg.detections:
            cx = d.bbox.center.position.x; cy = d.bbox.center.position.y
            w  = d.bbox.size_x; h  = d.bbox.size_y
            cls = None; score = None
            if d.results:
                cls = d.results[0].hypothesis.class_id
                score = float(d.results[0].hypothesis.score)
            dets.append({"cx":cx, "cy":cy, "w":w, "h":h, "class_id":cls, "score":score})
        with self._lock: self._det = {"num_detections":len(dets), "detections":dets}

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock: return self._jpeg

    def get_latest_detections(self) -> dict:
        with self._lock: return dict(self._det)

    def publish_movej(self, joints_rad: List[float], speed: float, accel: float, relative: bool):
        if len(joints_rad) != 6: raise ValueError("joints_rad must be length 6")
        data = list(joints_rad) + [float(speed), float(accel), 1.0 if relative else 0.0]
        m = Float64MultiArray(); m.data = data
        m.layout = MultiArrayLayout(dim=[
            MultiArrayDimension(label='joints', size=6, stride=9),
            MultiArrayDimension(label='extras', size=3, stride=3)], data_offset=0)
        self.pub_movej.publish(m)

    def publish_move_axis(self, axis: str, dist: float, speed: float | None, accel: float | None, relative: bool):
        ax = axis.upper()
        rel_flag = 1.0 if relative else 0.0
        spd = -1.0 if speed is None else float(speed)
        acc = -1.0 if accel is None else float(accel)
        if ax.startswith('J') and len(ax)==2 and ax[1].isdigit():
            idx = int(ax[1]) - 1
            if not (0 <= idx <= 5): raise ValueError("Joint axis must be J1..J6")
            m = Float64MultiArray(); m.data = [float(idx), float(dist), spd, acc, rel_flag]
            self.pub_move_axis.publish(m)
            q = [0.0]*6; q[idx]=float(dist); self.publish_movej(q, speed or 0.0, accel or 0.0, relative)
        else:
            code = {'X':0.0,'Y':1.0,'Z':2.0}.get(ax)
            if code is None: raise ValueError("AXIS must be J1..J6 or X/Y/Z")
            m = Float64MultiArray(); m.data = [code, float(dist), spd, acc, rel_flag]
            self.pub_move_xyz.publish(m)

    def publish_move_xyz(self, axis: str, dist: float, speed: float | None, accel: float | None, relative: bool):
        ax = axis.upper()
        code = {'X':0.0,'Y':1.0,'Z':2.0}.get(ax)
        if code is None: raise ValueError("AXIS must be X/Y/Z")
        rel_flag = 1.0 if relative else 0.0
        spd = -1.0 if speed is None else float(speed)
        acc = -1.0 if accel is None else float(accel)
        m = Float64MultiArray(); m.data = [code, float(dist), spd, acc, rel_flag]
        self.pub_move_xyz.publish(m)

    def publish_move_vision(self, axis: str, dist: float, speed: float | None, accel: float | None, relative: bool):
        ax = axis.upper()
        rel_flag = 1.0 if relative else 0.0
        spd = -1.0 if speed is None else float(speed)
        acc = -1.0 if accel is None else float(accel)
        if ax in ('X','Y','Z'):
            code = {'X':0.0,'Y':1.0,'Z':2.0}[ax]
        elif ax.startswith('J') and len(ax)==2 and ax[1].isdigit():
            code = float(int(ax[1])-1)
        else:
            raise ValueError("AXIS must be J1..J6 or X/Y/Z")
        m = Float64MultiArray(); m.data = [code, float(dist), spd, acc, rel_flag]
        self.pub_move_vision.publish(m)
