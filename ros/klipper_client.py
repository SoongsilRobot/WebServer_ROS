import requests
from typing import Optional

class KlipperClient:
    def __init__(self, base_url="http://127.0.0.1:7125", api_key: Optional[str]=None, timeout: float=3.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self._headers = {}
        if api_key: self._headers["X-Api-Key"] = api_key

    def _post(self, path: str, json_body: dict):
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=json_body, headers=self._headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def send_gcode(self, script: str):
        return self._post("/printer/gcode/script", {"script": script})

    def manual_stepper_move(self, stepper: str, move: float, speed: float=None, accel: float=None):
        cmd = f"MANUAL_STEPPER STEPPER={stepper} MOVE={move:.6f}"
        if speed is not None: cmd += f" SPEED={max(0.001,float(speed)):.3f}"
        if accel is not None: cmd += f" ACCEL={max(0.001,float(accel)):.3f}"
        return self.send_gcode(cmd)

    def manual_stepper_setpos(self, stepper: str, position: float):
        cmd = f"MANUAL_STEPPER STEPPER={stepper} SET_POSITION={position:.6f}"
        return self.send_gcode(cmd)

    def emergency_stop(self):
        return self.send_gcode("M112")

    def firmware_restart(self):
        return self.send_gcode("FIRMWARE_RESTART")

    def query_objects(self, objects: dict):
        """
        Moonraker /printer/objects/query
        objects 예:
        {
          "gcode_move": ["position", "speed"],
          "motion_report": ["live_position", "live_velocity", "steppers"],
          "stepper_enable": ["steppers"],
          "manual_stepper gear1": ["position", "enabled"]
        }
        """
        return self._post("/printer/objects/query", {"objects": objects})
