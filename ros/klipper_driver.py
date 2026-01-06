import json
import math
import time
from pathlib import Path

import rclpy
from concurrent.futures import ThreadPoolExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float64MultiArray, String
from .klipper_client import KlipperClient
from .kinematics import CartesianKinematics

try:
    import yaml
except Exception:
    yaml = None

_ROS_DIR = Path(__file__).resolve().parent
_DEFAULT_CFG = {
    "moonraker_url": "http://127.0.0.1:7125",
    "api_key": None,
    # 이동 명령용 조인트 스케일 (상태 수집과는 별개)
    "joints": [
        {"name": "J1", "stepper": "J1", "units_per_rad": 57.2957795, "speed_units_per_rads": 57.2957795},
        {"name": "J2", "stepper": "J2", "units_per_rad": 57.2957795, "speed_units_per_rads": 57.2957795},
        {"name": "J3", "stepper": "J3", "units_per_rad": 57.2957795, "speed_units_per_rads": 57.2957795},
        {"name": "J4", "stepper": "J4", "units_per_rad": 57.2957795, "speed_units_per_rads": 57.2957795},
        {"name": "J5", "stepper": "J5", "units_per_rad": 57.2957795, "speed_units_per_rads": 57.2957795},
        {"name": "J6", "stepper": "J6", "units_per_rad": 57.2957795, "speed_units_per_rads": 57.2957795},
    ],
    "deadband_rad": 1e-4,
    "enable_cartesian": True,
    "cartesian_linear_scale": 0.001,  # mm -> m
    "cartesian_angular_scale": math.pi / 180.0,  # deg -> rad
    "urdf_path": str(_ROS_DIR / "description" / "URDF_Setting.xacro"),
    "base_link": "base_link",
    "tip_link": "ee_link",
    "position_tolerance": 0.002,  # meters
    "orientation_tolerance": 0.05,  # rad
}

URDF_JOINT_ALIASES = {
    "J1": "joint1_base_yaw",
    "J2": "joint2_shoulder_pitch",
    "J3": "joint3_elbow_pitch",
    "J4": "joint4_wrist_roll",
    "J5": "joint5_wrist_pitch",
}

class KlipperDriver(Node):
    def __init__(self, config_path: str = "config/joints.yaml"):
        print("KlipperDriver Init")
        super().__init__('klipper_driver')
        self.declare_parameter('config_path', config_path)
        path = self.get_parameter('config_path').value
        axes = [0,0,0,0,0,0]
        cfg = dict(_DEFAULT_CFG)
        if yaml is None:
            self.get_logger().warn("yaml not available, using default config.")
            print("yaml not available, using default config.")
        else:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    for k in _DEFAULT_CFG:
                        if k in data:
                            cfg[k] = data[k]
            except Exception as e:
                self.get_logger().warn(f"Failed to read {path}, using defaults: {e}")

        cfg.setdefault("joint_aliases", URDF_JOINT_ALIASES)
        self.cfg = cfg
        self.deadband = float(cfg.get("deadband_rad", 1e-4))
        self.client = KlipperClient(cfg["moonraker_url"], cfg.get("api_key"))
        self.joints = cfg["joints"]
        self.axes = {f"J{i+1}": {"pos": 0.0, "speed": 0.0, "moving": False, "enabled": False} for i in range(6)}
        self.last_joint_targets = [0.0] * 6
        self.linear_scale = float(cfg.get("cartesian_linear_scale", 0.001))
        self.angular_scale = float(cfg.get("cartesian_angular_scale", math.pi / 180.0))
        self.enable_cartesian = bool(cfg.get("enable_cartesian", True))
        self.kinematics = None
        if self.enable_cartesian:
            xacro_path = Path(cfg.get("urdf_path", _DEFAULT_CFG["urdf_path"]))
            base_link = cfg.get("base_link", "base_link")
            tip_link = cfg.get("tip_link", "ee_link")
            aliases = cfg.get("joint_aliases", URDF_JOINT_ALIASES)
            pos_tol = float(cfg.get("position_tolerance", _DEFAULT_CFG["position_tolerance"]))
            ori_tol = float(cfg.get("orientation_tolerance", _DEFAULT_CFG["orientation_tolerance"]))
            try:
                self.kinematics = CartesianKinematics(
                    xacro_path,
                    base_link=base_link,
                    tip_link=tip_link,
                    joint_aliases=aliases,
                    position_tolerance=pos_tol,
                    orientation_tolerance=ori_tol,
                )
                self.last_joint_targets = self.kinematics.current_alias_joints()
                self.get_logger().info(f"Cartesian IK ready (model={xacro_path})")
            except Exception as exc:
                self.enable_cartesian = False
                self.get_logger().error(f"Failed to initialise kinematics: {exc}")
        self.cb_motion = ReentrantCallbackGroup()
        self.cb_status = ReentrantCallbackGroup()
        self._move_pool = ThreadPoolExecutor(max_workers=1)

        # Subscribers (move commands)
        self.sub_movej = self.create_subscription(Float64MultiArray, '/robot/movej_cmd', self.on_cmd, 10, callback_group=self.cb_motion)
        self.sub_axis  = self.create_subscription(Float64MultiArray, '/robot/move_axis_cmd', self.on_axis_cmd, 10, callback_group=self.cb_motion)
        self.sub_xyz   = self.create_subscription(Float64MultiArray, '/robot/move_xyz_cmd', self.on_xyz_cmd, 10, callback_group=self.cb_motion)
        self.sub_vis   = self.create_subscription(Float64MultiArray, '/robot/move_vision_cmd', self.on_vision_cmd, 10, callback_group=self.cb_motion)
        
        self.get_logger().info(f"KlipperDriver ready; Moonraker: {cfg['moonraker_url']}")

        # Publisher (status)
        self.pub_status = self.create_publisher(String, '/status', 10)

        # 속도 추정을 위한 이전 샘플(축값 + manual_stepper 값들)
        self._last = {"t": None, "x": None, "y": None, "z": None,
                      "gear1": None, "gear2": None, "gear3": None}

        # 5Hz 폴링
        self.create_timer(0.1, self._poll_status, callback_group=self.cb_status)

    # ---------------------- move command helpers ----------------------

    def _find_joint(self, name: str):
        for j in self.joints:
            if j.get("name") == name:
                return j
        return None

    def _send_stepper_delta(self, name: str, delta_rad: float, speed_rads: float, accel_rads2: float):
        """
        현재 구현은 MANUAL_STEPPER 기반으로 만들어져 있습니다.
        (주의) X/Y/Z 같은 기본 stepper를 직접 수동 구동하려면 별도 G-code가 필요합니다.
        지금은 J4~J6(gear1..3) 위주 사용을 가정합니다.
        """
        print("SEND delta")
        j = self._find_joint(name)
        if j is None:
            self.get_logger().warn(f"No mapping for axis '{name}' in config; skipping.")
            return
        
        units_per_rad = float(j.get("units_per_rad", 1))
        speed_units_per_rads = float(j.get("speed_units_per_rads", units_per_rad))
        stepper = j.get("stepper", name)

        move_units = delta_rad * units_per_rad
        
        speed_units = None
        if (speed_rads is not None) and (speed_rads != 0):
            speed_units = max(0.001, speed_rads * speed_units_per_rads)

        # HTTP 요청이 길어도 ROS 콜백을 막지 않도록 전용 쓰레드에서 실행
        self._move_pool.submit(self._execute_stepper_move, stepper, move_units, speed_units, accel_rads2, delta_rad)

    def _execute_stepper_move(self, stepper: str, move_units: float, speed_units: float | None,
                              accel_rads2: float, delta_rad: float):
        try:
            self.client.manual_stepper_move(stepper=stepper, move=move_units, speed=speed_units, accel=accel_rads2)
            self.get_logger().info(f"[{stepper}] MOVE={move_units:.4f} (dq={delta_rad:.4f} rad)")
            print(f"[{stepper}] MOVE={move_units:.4f} (dq={delta_rad:.4f} rad)")
        except Exception as e:
            self.get_logger().error(f"Moonraker request failed: {e}")
            print(f"Moonraker request failed: {e}")

    def _dispatch_movej(self, joint_deltas, speed, accel):
        payload = list(joint_deltas)
        if len(payload) < 6:
            payload.extend([0.0] * (6 - len(payload)))
        spd = -1.0 if speed is None else float(speed)
        acc = -1.0 if accel is None else float(accel)
        msg = Float64MultiArray()
        msg.data = payload[:6] + [spd, acc, 1.0]
        self.on_cmd(msg)

    # ---------------------- subscribers ----------------------

    def on_cmd(self, msg: Float64MultiArray):        
        d = list(msg.data)
        print("on_cmd's d:",d)
        if len(d) < 9:
            self.get_logger().warn("movej_cmd invalid")
            return
        
        q = list(d[:6])
        if len(q) < 6:
            q.extend([0.0] * (6 - len(q)))
        print("Before q: " ,q)
        speed = None if d[6] == -1.0 else float(d[6])
        accel = None if d[7] == -1.0 else float(d[7])
        rel = int(d[8])
        if rel == 0:
            self.get_logger().warn("Absolute requested; treating as relative")
            cached_pos = [self.axes.get(f"J{i+1}", {}).get("pos", 0.0) for i in range(len(q))]
            q_rel = []
            for target, cur in zip(q, cached_pos):
                if abs(target) < self.deadband:
                    q_rel.append(0.0)
                else:
                    q_rel.append(target - cur)
            q = q_rel
            print("Now q (converted to relative deltas): ", q)
        # Lst = ['X','Y','Z','A','B','C']
        for i, dq in enumerate(q):
            dq = float(dq)
            if abs(dq) < self.deadband:
                continue
            self._send_stepper_delta(f"J{i+1}", dq, speed, accel)

    def on_axis_cmd(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) < 5:
            self.get_logger().warn("axis_cmd invalid")
            return
        idx = int(d[0])
        dq = float(d[1])
        speed = float(d[2]) if d[2] != -1.0 else None
        accel = float(d[3]) if d[3] != -1.0 else None
        self._send_stepper_delta(f"J{idx+1}", dq, speed, accel)

    def on_xyz_cmd(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) < 5:
            self.get_logger().warn("xyz_cmd invalid")
            return
        code = int(d[0])
        dist = float(d[1])
        speed = None if d[2] == -1.0 else float(d[2])
        accel = None if d[3] == -1.0 else float(d[3])
        rel_flag = int(d[4]) if len(d) > 4 else 1
        axis_map = {0: "X", 1: "Y", 2: "Z", 3: "YAW", 4: "P"}
        axis_name = axis_map.get(code)
        if axis_name is None:
            self.get_logger().warn("xyz_cmd unknown axis")
            return
        if not self.kinematics:
            self.get_logger().warn("Cartesian command received but IK is disabled")
            return
        scale = self.linear_scale if axis_name in ("X", "Y", "Z") else self.angular_scale
        mode = "relative" if rel_flag == 1 else "absolute"
        try:
            joints_abs = self.kinematics.apply_axis_command(axis_name, dist * scale, mode)
        except Exception as exc:
            self.get_logger().error(f"IK solve failed for axis {axis_name}: {exc}")
            return
        rel_joints = [new - old for new, old in zip(joints_abs, self.last_joint_targets)]
        self.last_joint_targets = list(joints_abs)
        self._dispatch_movej(rel_joints, speed, accel)

    def on_vision_cmd(self, msg: Float64MultiArray):
        # currently same as xyz
        self.on_xyz_cmd(msg)

    # ---------------------- status polling ----------------------

    def _poll_status(self):
        def fnum(v, default=0.0):
            try:
                if v is None:
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        def deriv(cur, prev, dt):
            if prev is None or not dt or dt <= 0:
                return 0.0
            return (cur - prev) / dt

        def get_macro_var(status_dict, macro_name: str, var_name: str):
            """
            gcode_macro 객체는 환경에 따라
            - status["gcode_macro MACRO"][var_name]
            - status["gcode_macro MACRO"]["variables"][var_name]
            두 형태가 존재한다. 둘 다 시도.
            """
            obj = status_dict.get(f"gcode_macro {macro_name}", {}) or {}
            # flat 형태
            if isinstance(obj, dict) and var_name in obj:
                return obj.get(var_name)
            # variables 맵
            vars_map = obj.get("variables", {}) if isinstance(obj, dict) else {}
            if isinstance(vars_map, dict):
                return vars_map.get(var_name)
            return None

        try:
            objs = {
                "toolhead": ["position", "homed_axes", "max_velocity", "max_accel"],
                "gcode_move": ["gcode_position", "speed", "homing_origin"],
                "motion_report": ["live_position", "live_velocity"],
                "stepper_enable": ["steppers"],
                # manual_stepper (J4~J6)
                "manual_stepper gear1": ["position", "enabled"],
                "manual_stepper gear2": ["position", "enabled"],
                "manual_stepper gear3": ["position", "enabled"],
                # 매크로 변수 폴백(둘 다 질의해 두면 이름 달라도 커버)
                "gcode_macro STATE_G1": ["pos"],
                "gcode_macro STATE_G2": ["pos"],
                "gcode_macro STATE_G3": ["pos"],
                "gcode_macro GEAR1_STATE": ["pos"],
                "gcode_macro GEAR2_STATE": ["pos"],
                "gcode_macro GEAR3_STATE": ["pos"],
            }
            res = self.client.query_objects(objs)
            status = (res or {}).get("result", {}).get("status", {})
        except Exception as e:
            self.get_logger().warn(f"status query failed: {e}")
            return

        now = time.time()

        # toolhead / gcode_move
        tool = status.get("toolhead", {}) or {}
        th_pos = tool.get("position") or [0, 0, 0, 0]
        th_homed = tool.get("homed_axes", "")
        th_vmax = fnum(tool.get("max_velocity"))
        th_amax = fnum(tool.get("max_accel"))

        gcm = status.get("gcode_move", {}) or {}
        self.gpos = gcm.get("gcode_position") or [0, 0, 0, 0,0]
        # for i in (5-self.gpos.count()):
        #     self.gpos.append(0)
        
        g_speed = fnum(gcm.get("speed"))
        homing_origin = gcm.get("homing_origin") or [0, 0, 0, 0]

        # XYZ 위치/속도: live_position 우선, 없으면 toolhead.position
        mp = status.get("motion_report", {}) or {}
        base_pos = (mp.get("live_position") or th_pos)
        x, y, z = fnum(base_pos[0]), fnum(base_pos[1]), fnum(base_pos[2])

        dt = (now - self._last["t"]) if self._last["t"] else None
        vx = deriv(x, self._last.get("x"), dt)
        vy = deriv(y, self._last.get("y"), dt)
        vz = deriv(z, self._last.get("z"), dt)
        lv = fnum(mp.get("live_velocity"))

        # manual_stepper 원본
        m1 = status.get("manual_stepper gear1", {}) or {}
        m2 = status.get("manual_stepper gear2", {}) or {}
        m3 = status.get("manual_stepper gear3", {}) or {}

        # 매크로 변수 폴백 (STATE_G* 우선, 없으면 GEAR*_STATE)
        fb1 = get_macro_var(status, "STATE_G1", "pos")
        if fb1 is None:
            fb1 = get_macro_var(status, "GEAR1_STATE", "pos")
        fb2 = get_macro_var(status, "STATE_G2", "pos")
        if fb2 is None:
            fb2 = get_macro_var(status, "GEAR2_STATE", "pos")
        fb3 = get_macro_var(status, "STATE_G3", "pos")
        if fb3 is None:
            fb3 = get_macro_var(status, "GEAR3_STATE", "pos")

        # pos: manual_stepper.position 우선 → 없으면 매크로 변수 pos
        g1 = fnum(fb1 if fb1 is not None else m1.get("position", fb1))
        g2 = fnum(fb2 if fb2 is not None else m2.get("position", fb2))
        g3 = fnum(fb3 if fb3 is not None else m3.get("position", fb3))

        # speed/moving: Δpos/Δt
        vg1 = deriv(g1, self._last.get("gear1"), dt)
        vg2 = deriv(g2, self._last.get("gear2"), dt)
        vg3 = deriv(g3, self._last.get("gear3"), dt)

        en_map = status.get("stepper_enable", {}).get("steppers", {}) or {}
        en_g1 = bool(m1.get("enabled", en_map.get("gear1", False)))
        en_g2 = bool(m2.get("enabled", en_map.get("gear2", False)))
        en_g3 = bool(m3.get("enabled", en_map.get("gear3", False)))

        # 정지/동작 판정 임계값 (필요시 조정)
        EPS = 1e-3  # mm/s

        self.axes = {
            "J1": {"pos": x, "speed": vx, "moving": abs(vx) > EPS,
                   "enabled": bool(en_map.get("stepper_x", False))},
            "J2": {"pos": y, "speed": vy, "moving": abs(vy) > EPS,
                   "enabled": bool(en_map.get("stepper_y", False))},
            "J3": {"pos": z, "speed": vz, "moving": abs(vz) > EPS,
                   "enabled": bool(en_map.get("stepper_z", False))},
            "J4": {"pos": g1, "speed": vg1, "moving": abs(vg1) > EPS, "enabled": en_g1},
            "J5": {"pos": g2, "speed": vg2, "moving": abs(vg2) > EPS, "enabled": en_g2},
            "J6": {"pos": g3, "speed": vg3, "moving": abs(vg3) > EPS, "enabled": en_g3},
        }

        payload = {
            "timestamp": now,
            "axes": self.axes,
            "toolhead": {
                "position": th_pos,
                "homed_axes": th_homed,
                "max_velocity": th_vmax,
                "max_accel": th_amax,
            },
            "gcode_move": {
                "gcode_position": self.gpos,
                "speed": g_speed,
                "homing_origin": homing_origin,
            },
            "live_velocity": lv,
            "source": "moonraker",
        }

        # 캐시 업데이트 (미분에 쓴 동일 키!)
        self._last.update({"t": now, "x": x, "y": y, "z": z,
                           "gear1": g1, "gear2": g2, "gear3": g3})

        # 디버그(필요시 주석 해제)
        # self.get_logger().info(
        #     f"J4 src={'ms' if 'position' in (m1 or {}) else 'fb'} pos={g1} "
        #     f"STATE_G1={get_macro_var(status,'STATE_G1','pos')} "
        #     f"GEAR1_STATE={get_macro_var(status,'GEAR1_STATE','pos')}"
        # )

        # 퍼블리시
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KlipperDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
