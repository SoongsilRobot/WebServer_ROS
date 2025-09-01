import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from .klipper_client import KlipperClient
try: import yaml
except Exception: yaml = None

_DEFAULT_CFG = {
    "moonraker_url": "http://127.0.0.1:7125",
    "api_key": None,
    "joints": [
        {"name":"J1","stepper":"J1","units_per_rad":57.2957795,"speed_units_per_rads":57.2957795},
        {"name":"J2","stepper":"J2","units_per_rad":57.2957795,"speed_units_per_rads":57.2957795},
        {"name":"J3","stepper":"J3","units_per_rad":57.2957795,"speed_units_per_rads":57.2957795},
        {"name":"J4","stepper":"J4","units_per_rad":57.2957795,"speed_units_per_rads":57.2957795},
        {"name":"J5","stepper":"J5","units_per_rad":57.2957795,"speed_units_per_rads":57.2957795},
        {"name":"J6","stepper":"J6","units_per_rad":57.2957795,"speed_units_per_rads":57.2957795},
    ],
    "deadband_rad": 1e-4
}

class KlipperDriver(Node):
    def __init__(self, config_path: str = "config/joints.yaml"):
        super().__init__('klipper_driver')
        self.declare_parameter('config_path', config_path)
        path = self.get_parameter('config_path').value
        cfg = dict(_DEFAULT_CFG)
        if yaml is None:
            self.get_logger().warn("yaml not available, using default config.")
        else:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    for k in _DEFAULT_CFG:
                        if k in data: cfg[k] = data[k]
            except Exception as e:
                self.get_logger().warn(f"Failed to read {path}, using defaults: {e}")

        self.cfg = cfg
        self.deadband = float(cfg.get("deadband_rad", 1e-4))
        self.client = KlipperClient(cfg["moonraker_url"], cfg.get("api_key"))
        self.joints = cfg["joints"]

        self.create_subscription(Float64MultiArray, '/robot/movej_cmd', self.on_cmd, 10)
        self.create_subscription(Float64MultiArray, '/robot/move_axis_cmd', self.on_axis_cmd, 10)
        self.create_subscription(Float64MultiArray, '/robot/move_xyz_cmd', self.on_xyz_cmd, 10)
        self.create_subscription(Float64MultiArray, '/robot/move_vision_cmd', self.on_vision_cmd, 10)

        self.get_logger().info(f"KlipperDriver ready; Moonraker: {cfg['moonraker_url']}")

    def _find_joint(self, name: str):
        for j in self.joints:
            if j.get("name")==name: return j
        return None

    def _send_stepper_delta(self, name: str, delta_rad: float, speed_rads: float, accel_rads2: float):
        j = self._find_joint(name)
        if j is None:
            self.get_logger().warn(f"No mapping for axis '{name}' in config; skipping."); return
        units_per_rad = float(j.get("units_per_rad", 57.2957795))
        speed_units_per_rads = float(j.get("speed_units_per_rads", units_per_rad))
        stepper = j.get("stepper", name)
        move_units = delta_rad * units_per_rad
        speed_units = max(0.001, speed_rads * speed_units_per_rads)
        try:
            self.client.manual_stepper_move(stepper=stepper, move=move_units, speed=speed_units, accel=accel_rads2)
            self.get_logger().info(f"[{stepper}] MOVE={move_units:.4f} (dq={delta_rad:.4f} rad)")
        except Exception as e:
            self.get_logger().error(f"Moonraker request failed: {e}")

    def on_cmd(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) < 9: self.get_logger().warn("movej_cmd invalid"); return
        q = d[:6]; speed = float(d[6]); accel = float(d[7]); rel = int(d[8])
        if rel == 0: self.get_logger().warn("Absolute requested; treating as relative")
        for i, dq in enumerate(q):
            dq = float(dq)
            if abs(dq) < self.deadband: continue
            self._send_stepper_delta(f"J{i+1}", dq, speed, accel)

    def on_axis_cmd(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) < 5: self.get_logger().warn("axis_cmd invalid"); return
        idx = int(d[0]); dq = float(d[1]); speed = float(d[2]); accel = float(d[3])
        self._send_stepper_delta(f"J{idx+1}", dq, speed, accel)

    def on_xyz_cmd(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) < 5: self.get_logger().warn("xyz_cmd invalid"); return
        code = int(d[0]); dist = float(d[1]); speed = float(d[2]); accel = float(d[3])
        axis_map = {0:"X",1:"Y",2:"Z"}
        name = axis_map.get(code, None)
        if name is None: self.get_logger().warn("xyz_cmd unknown axis"); return
        self._send_stepper_delta(name, dist, speed, accel)

    def on_vision_cmd(self, msg: Float64MultiArray):
        # currently same as xyz
        self.on_xyz_cmd(msg)
