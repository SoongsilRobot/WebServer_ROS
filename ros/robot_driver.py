import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
try: import serial
except Exception: serial = None

class RobotDriver(Node):
    def __init__(self, use_serial=False, serial_port='/dev/ttyACM0', baud=115200):
        super().__init__('robot_driver')
     #   self.use_serial = bool(use_serial); self.serial_port = serial_port; self.baud = int(baud)
     #   self.ser = None
     #   if self.use_serial and serial is not None:
     #       try:
     #           self.ser = serial.Serial(self.serial_port, self.baud, timeout=0.1)
     #           self.get_logger().info(f"Serial opened: {self.serial_port} @ {self.baud}")
     #       except Exception as e:
     #           self.get_logger().error(f"Serial open failed: {e}")
        # self.create_subscription(Float64MultiArray, '/robot/movej_cmd', self.on_cmd, 10)

    #def on_cmd(self, msg: Float64MultiArray):
    #    d = list(msg.data)
    #    if len(d) < 9: self.get_logger().warn("invalid movej payload"); return
    #    j=d[:6]; speed=d[6]; accel=d[7]; rel=d[8]
    #    self.get_logger().info(f"MoveJ: {j} speed={speed} accel={accel} relative={bool(rel)}")
    #    if self.ser:
    #        try:
    #            line = "MOVEJ " + " ".join(f"{x:.6f}" for x in j) + f" SPEED {speed:.3f} ACC {accel:.3f} REL {int(rel)}\n"
    #            self.ser.write(line.encode('ascii'))
    #        except Exception as e:
    #            self.get_logger().error(f"Serial write failed: {e}")
