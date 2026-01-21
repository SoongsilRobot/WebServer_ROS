import math
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple


L1 = 0.13461
L2 = 0.0475
L3 = 0.075
L4 = 0.24168
TOOL = 0.08597
BASE_Z = 0.02

JOINT_MIN = -math.pi
JOINT_MAX = math.pi


def _load_params_from_xacro(xacro_path: str) -> None:
    if not os.path.exists(xacro_path):
        return
    try:
        tree = ET.parse(xacro_path)
        root = tree.getroot()
        props = {}
        for elem in root.findall(".//{*}property"):
            name = elem.attrib.get("name")
            value = elem.attrib.get("value")
            if name and value:
                try:
                    props[name] = float(value)
                except ValueError:
                    continue
        globals_map = globals()
        for key in ("L1", "L2", "L3", "L4", "TOOL"):
            if key in props:
                globals_map[key] = props[key]
    except ET.ParseError:
        return


_load_params_from_xacro(os.path.join(os.path.dirname(__file__), "robot.xacro"))


def _wrap_angle(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    res = [[0.0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            aik = a[i][k]
            for j in range(len(b[0])):
                res[i][j] += aik * b[k][j]
    return res


def _mat_vec_mul(a: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(a))]


def _mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _mat_identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def _invert_matrix(a: List[List[float]]) -> Optional[List[List[float]]]:
    n = len(a)
    aug = [row[:] + ident_row[:] for row, ident_row in zip(a, _mat_identity(n))]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if abs(aug[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for c in range(2 * n):
            aug[col][c] /= pivot_val
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for c in range(2 * n):
                aug[r][c] -= factor * aug[col][c]
    return [row[n:] for row in aug]


def _rotz(theta: float) -> List[List[float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _roty(theta: float) -> List[List[float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotx(theta: float) -> List[List[float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _transl(x: float, y: float, z: float) -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _fk(q: List[float]) -> Tuple[List[float], List[List[float]]]:
    q1, q2, q3, q4, q5 = q
    t = _transl(0.0, 0.0, BASE_Z)
    t = _mat_mul(t, _rotz(q1))
    t = _mat_mul(t, _transl(0.0, 0.0, L1))
    t = _mat_mul(t, _roty(q2))
    t = _mat_mul(t, _transl(L2, 0.0, 0.0))
    t = _mat_mul(t, _roty(q3))
    t = _mat_mul(t, _transl(L3, 0.0, 0.0))
    t = _mat_mul(t, _rotx(q4))
    t = _mat_mul(t, _transl(L4, 0.0, 0.0))
    t = _mat_mul(t, _roty(q5))
    t = _mat_mul(t, _transl(TOOL, 0.0, 0.0))
    pos = [t[0][3], t[1][3], t[2][3]]
    rot = [row[:3] for row in t[:3]]
    return pos, rot


def _roll_pitch_from_rot(rot: List[List[float]]) -> Tuple[float, float]:
    r20 = rot[2][0]
    r21 = rot[2][1]
    r22 = rot[2][2]
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    roll = math.atan2(r21, r22)
    return roll, pitch


def _fk_features(q: List[float]) -> List[float]:
    pos, rot = _fk(q)
    roll, pitch = _roll_pitch_from_rot(rot)
    return [pos[0], pos[1], pos[2], pitch, roll]


def _numeric_jacobian(q: List[float], eps: float = 1e-6) -> List[List[float]]:
    base = _fk_features(q)
    j = [[0.0] * 5 for _ in range(5)]
    for i in range(5):
        q_eps = q[:]
        q_eps[i] += eps
        f_eps = _fk_features(q_eps)
        for r in range(5):
            diff = f_eps[r] - base[r]
            if r >= 3:
                diff = _wrap_angle(diff)
            j[r][i] = diff / eps
    return j


def solve_ik(
    x: float,
    y: float,
    z: float,
    pitch: float,
    roll: float,
    max_iters: int = 200,
    pos_tol: float = 1e-4,
    ang_tol: float = 1e-3,
) -> Optional[List[float]]:
    seeds = []
    base_yaw = math.atan2(y, x) if (abs(x) + abs(y)) > 1e-9 else 0.0
    seeds.append([base_yaw, 0.0, 0.0, roll, pitch])
    seeds.append([base_yaw, 0.5, -0.5, roll, pitch])
    seeds.append([base_yaw, -0.5, 0.5, roll, pitch])

    target = [x, y, z, pitch, roll]

    for seed in seeds:
        q = seed[:]
        for _ in range(max_iters):
            f = _fk_features(q)
            err = [
                target[0] - f[0],
                target[1] - f[1],
                target[2] - f[2],
                _wrap_angle(target[3] - f[3]),
                _wrap_angle(target[4] - f[4]),
            ]
            if (
                math.sqrt(err[0] ** 2 + err[1] ** 2 + err[2] ** 2) < pos_tol
                and math.sqrt(err[3] ** 2 + err[4] ** 2) < ang_tol
            ):
                return [ _wrap_angle(v) for v in q ]
            j = _numeric_jacobian(q)
            jt = _mat_transpose(j)
            jj_t = _mat_mul(j, jt)
            damp = 1e-4
            jj_t = _mat_add(jj_t, [[damp if i == j else 0.0 for j in range(5)] for i in range(5)])
            inv = _invert_matrix(jj_t)
            if inv is None:
                break
            step = _mat_vec_mul(_mat_mul(jt, inv), err)
            for i in range(5):
                q[i] = max(JOINT_MIN, min(JOINT_MAX, q[i] + 0.6 * step[i]))
    return None


class CartesianKinematics:
    def __init__(
        self,
        xacro_path: str,
        base_link: str = "base_link",
        tip_link: str = "ee_link",
        joint_aliases: Optional[dict] = None,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-2,
    ) -> None:
        _load_params_from_xacro(str(xacro_path))
        self.base_link = base_link
        self.tip_link = tip_link
        self.joint_aliases = joint_aliases or {
            "J1": "joint1_base_yaw",
            "J2": "joint2_shoulder_pitch",
            "J3": "joint3_elbow_pitch",
            "J4": "joint4_wrist_roll",
            "J5": "joint5_wrist_pitch",
        }
        self.alias_order = [k for k in ("J1", "J2", "J3", "J4", "J5") if k in self.joint_aliases]
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self._joints = [0.0] * 5
        self._target = _fk_features(self._joints)

    def current_alias_joints(self) -> List[float]:
        return [self._joints[self._alias_index(name)] for name in self.alias_order]

    def apply_axis_command(self, axis_name: str, dist: float, mode: str = "relative") -> List[float]:
        axis = axis_name.upper()
        if mode not in ("relative", "absolute"):
            raise ValueError(f"Unknown mode '{mode}'")
        if axis in ("YAW", "RZ"):
            joint_idx = self._alias_index("J1")
            if mode == "relative":
                self._joints[joint_idx] = _wrap_angle(self._joints[joint_idx] + dist)
            else:
                self._joints[joint_idx] = _wrap_angle(dist)
            self._target = _fk_features(self._joints)
            return self.current_alias_joints()

        target = list(self._target)
        if axis == "X":
            target[0] = target[0] + dist if mode == "relative" else dist
        elif axis == "Y":
            target[1] = target[1] + dist if mode == "relative" else dist
        elif axis == "Z":
            target[2] = target[2] + dist if mode == "relative" else dist
        elif axis in ("P", "PITCH", "RY"):
            target[3] = _wrap_angle(target[3] + dist) if mode == "relative" else _wrap_angle(dist)
        else:
            raise ValueError(f"Unknown axis '{axis_name}'")

        sol = solve_ik(
            target[0],
            target[1],
            target[2],
            target[3],
            target[4],
            pos_tol=self.position_tolerance,
            ang_tol=self.orientation_tolerance,
        )
        if sol is None:
            raise ValueError("IK solve failed for requested move")
        self._joints = list(sol)
        self._target = _fk_features(self._joints)
        return self.current_alias_joints()

    def apply_pose_command(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        yaw: float,
        pitch: float,
        mode: str = "relative",
    ) -> List[float]:
        if mode not in ("relative", "absolute"):
            raise ValueError(f"Unknown mode '{mode}'")
        target = list(self._target)
        if mode == "relative":
            target[0] += x
            target[1] += y
            target[2] += z
            target[3] = _wrap_angle(target[3] + pitch)
            target[4] = _wrap_angle(target[4] + roll)
        else:
            target[0] = x
            target[1] = y
            target[2] = z
            target[3] = _wrap_angle(pitch)
            target[4] = _wrap_angle(roll)

        sol = solve_ik(
            target[0],
            target[1],
            target[2],
            target[3],
            target[4],
            pos_tol=self.position_tolerance,
            ang_tol=self.orientation_tolerance,
        )
        if sol is None:
            raise ValueError("IK solve failed for requested pose")
        self._joints = list(sol)
        self._target = _fk_features(self._joints)
        return self.current_alias_joints()

    def _alias_index(self, alias: str) -> int:
        if alias == "J1":
            return 0
        if alias == "J2":
            return 1
        if alias == "J3":
            return 2
        if alias == "J4":
            return 3
        if alias == "J5":
            return 4
        raise ValueError(f"Unknown joint alias '{alias}'")


def move_to(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
) -> Optional[List[float]]:
    return solve_ik(x, y, z, pitch, roll)


def main() -> None:
    import sys

    if len(sys.argv) != 6:
        print("Usage: python ik_solver.py x y z roll pitch")
        sys.exit(2)
    x, y, z, roll, pitch = map(float, sys.argv[1:])
    sol = move_to(x, y, z, roll, pitch)
    if sol is None:
        print("false")
        sys.exit(1)
    print(sol)


if __name__ == "__main__":
    main()
