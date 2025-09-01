import argparse, threading
import uvicorn
import rclpy
from rclpy.executors import MultiThreadedExecutor
from fast_server.app import create_app
from ros.cam_publisher import CamPublisher
from ros.yolo_node import YoloNode
from ros.bridge import ROSBridge
from ros.robot_driver import RobotDriver
from ros.klipper_driver import KlipperDriver

def start_fastapi(app, host, port):
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config); server.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0'); parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device_index', type=int, default=0)
    parser.add_argument('--width', type=int, default=640); parser.add_argument('--height', type=int, default=480); parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--onnx_path', default='ros/models/yolov8n.onnx'); parser.add_argument('--input_size', type=int, default=320)
    parser.add_argument('--conf_thres', type=float, default=0.25); parser.add_argument('--onnx_format', default='auto')
    parser.add_argument('--debug_draw_letterbox', action='store_true')
    parser.add_argument('--use_serial', action='store_true'); parser.add_argument('--serial_port', default='/dev/ttyACM0'); parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--use_klipper', action='store_true'); parser.add_argument('--klipper_config', default='config/joints.yaml')
    args = parser.parse_args()

    rclpy.init()
    cam = CamPublisher(args.device_index, args.width, args.height, args.fps, '/vision/streaming/raw')
    yolo = YoloNode('/vision/streaming/raw','/vision/streaming','/vision/detections', args.onnx_path, args.input_size, args.conf_thres, args.onnx_format, args.debug_draw_letterbox)
    bridge = ROSBridge()
    driver = RobotDriver(args.use_serial, args.serial_port, args.baud)
    klipper = KlipperDriver(args.klipper_config) if args.use_klipper else None

    app = create_app(bridge, cors_origins=['*'])

    executor = MultiThreadedExecutor()
    nodes = [cam, yolo, bridge, driver] + ([klipper] if klipper else [])
    for n in nodes: executor.add_node(n)
    ros_thread = threading.Thread(target=executor.spin, daemon=True); ros_thread.start()

    start_fastapi(app, args.host, args.port)

    for n in nodes: n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
