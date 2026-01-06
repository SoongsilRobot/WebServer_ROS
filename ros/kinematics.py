import math
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import xacro  # type: ignore
    from ikpy.chain import Chain  # type: ignore
    _KIN_IMPORT_ERR = None
except Exception as exc:  # pragma: no cover
    xacro = None  # type: ignore
    Chain = None  # type: ignore
    _KIN_IMPORT_ERR = exc


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class CartesianKinematics:
    """Utility that keeps a cartesian pose target and resolves IK via ikpy."""

    def __init__(
        self,
        xacro_path: str | Path,
        base_link: str = "base_link",
        tip_link: str = "ee_link",
        joint_aliases: Dict[str, str] | None = None,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 5e-2,
    ):
        if _KIN_IMPORT_ERR:
            raise RuntimeError(
                "xacro/ikpy are required for CartesianKinematics"
            ) from _KIN_IMPORT_ERR
        self.xacro_path = Path(xacro_path)
        if not self.xacro_path.exists():
            raise FileNotFoundError(f"URDF/Xacro file not found: {self.xacro_path}")
        self.base_link = base_link
        self.tip_link = tip_link
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.joint_aliases = joint_aliases or {}

        self._chain = self._load_chain()
        self._link_index = {link.name: idx for idx, link in enumerate(self._chain.links)}
        self._current_full = np.zeros(len(self._chain.links))
        self._last_alias = [0.0] * 6  # default MoveJ payload size
        self._cart_state = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}
        self._refresh_cart_state()

    # ----------------------- public API -----------------------

    def current_alias_joints(self) -> List[float]:
        return list(self._last_alias)

    def apply_axis_command(self, axis: str, value: float, mode: str) -> List[float]:
        """Update the stored cartesian target and run IK. Returns absolute joint targets."""
        norm_axis = axis.upper()
        key = self._axis_to_key(norm_axis)
        prev_state = dict(self._cart_state)
        if mode == "relative":
            self._cart_state[key] += value
        else:
            self._cart_state[key] = value
        try:
            joints = self._solve()
        except Exception:
            # rollback cartesian state on failure
            self._cart_state = prev_state
            raise
        return joints

    # ----------------------- internal helpers -----------------------

    def _axis_to_key(self, axis: str) -> str:
        if axis in ("X", "Y", "Z"):
            return axis.lower()
        if axis in ("YAW", "RZ"):
            return "yaw"
        if axis in ("P", "PITCH", "RY"):
            return "pitch"
        raise ValueError(f"Unsupported cartesian axis '{axis}'")

    def _load_chain(self) -> Chain:
        xml = xacro.process_file(str(self.xacro_path)).toxml()
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as tmp:
            tmp.write(xml)
            tmp_path = Path(tmp.name)

        base_elements = [self.base_link] if self.base_link else None
        # First load to inspect link types
        chain_all = Chain.from_urdf_file(str(tmp_path), base_elements=base_elements)
        mask = []
        for idx, link in enumerate(chain_all.links):
            active = link.joint_type != "fixed"
            if idx == 0:
                active = False
            if link.name == self.tip_link:
                active = False
            if active and link.joint_type == "revolute":
                # loosen bounds to allow IK to explore both directions
                link.bounds = (-math.tau, math.tau)
            mask.append(active)
        return Chain.from_urdf_file(
            str(tmp_path), base_elements=base_elements, active_links_mask=mask
        )

    def _refresh_cart_state(self):
        fk = self._chain.forward_kinematics(self._current_full)
        self._cart_state["x"], self._cart_state["y"], self._cart_state["z"] = fk[:3, 3]
        self._cart_state["yaw"] = math.atan2(fk[1, 0], fk[0, 0])
        self._cart_state["pitch"] = math.atan2(-fk[2, 0], math.sqrt(fk[2, 1] ** 2 + fk[2, 2] ** 2))
        self._last_alias = self._full_to_alias(self._current_full)

    def _target_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = np.array([self._cart_state["x"], self._cart_state["y"], self._cart_state["z"]])
        T[:3, :3] = _rot_z(self._cart_state["yaw"]) @ _rot_y(self._cart_state["pitch"])
        return T

    def _solve(self) -> List[float]:
        target = self._target_matrix()
        solution = self._chain.inverse_kinematics_frame(
            target, initial_position=self._current_full, orientation_mode="all"
        )
        fk = self._chain.forward_kinematics(solution)
        pos_err = np.linalg.norm(fk[:3, 3] - target[:3, 3])
        ori_err = np.linalg.norm(fk[:3, :3] - target[:3, :3])
        if pos_err > self.position_tolerance:
            raise RuntimeError(f"IK position error too large ({pos_err:.4f} m)")
        if ori_err > self.orientation_tolerance:
            raise RuntimeError(f"IK orientation error too large ({ori_err:.4f})")
        self._current_full = solution
        self._refresh_cart_state()
        return self.current_alias_joints()

    def _full_to_alias(self, joints_full: Sequence[float]) -> List[float]:
        output = []
        for alias in ("J1", "J2", "J3", "J4", "J5", "J6"):
            urdf_name = self.joint_aliases.get(alias)
            if not urdf_name:
                output.append(0.0)
                continue
            idx = self._link_index.get(urdf_name)
            if idx is None:
                raise KeyError(f"Joint '{urdf_name}' not present in URDF chain")
            output.append(float(joints_full[idx]))
        return output
