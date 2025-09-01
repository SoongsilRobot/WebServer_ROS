import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CamPublisher(Node):
    def __init__(self, device_index=0, width=640, height=480, fps=10, topic='/vision/streaming/raw'):
        super().__init__('cam_publisher')
        self.bridge = CvBridge()
        qos_sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Image, topic, qos_sensor)
        self.cap = cv2.VideoCapture(int(device_index))
        if width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:    self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        self.timer = self.create_timer(1.0/max(int(fps),1), self._tick)

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None: return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub.publish(msg)
        
    def destroy_node(self):
        try: self.cap.release()
        except Exception: pass
        super().destroy_node()
