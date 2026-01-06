from pydantic import BaseModel, field_validator
from typing import Optional, List
import asyncio, re

class MoveJBody(BaseModel):
    joints: List[float]
    speed: float = 0.5
    accel: float = 1.0
    relative: bool = False
    @field_validator('joints')
    @classmethod
    def _len6(cls, v):
        if len(v) != 6:
            raise ValueError("joints must be length 6")
        return v

class AxisMoveBody(BaseModel):
    AXIS: str
    DIST: float
    SPD: float | None = None
    ACC: float | None = None
    MODE: str = "relative"
    @field_validator('MODE')
    @classmethod
    def _mode_ok(cls, v):
        v2 = str(v).strip().lower()
        if v2 not in ("relative","absolute"):
            raise ValueError("MODE must be 'relative' or 'absolute'")
        return v2
    @field_validator('AXIS')
    @classmethod
    def _axis_ok(cls, v):
        v2 = str(v).strip().upper()
        ok = (re.match(r'^J[1-6]$', v2) is not None) or (v2 in ("X","Y","Z"))
        if not ok:
            raise ValueError("AXIS must be J1..J6 or X/Y/Z")
        return v2
    class Config:
        json_schema_extra = {"example":{"AXIS":"J1","DIST":100.0,"SPD":0.5,"ACC":1.0,"MODE":"relative"}}

class XYZMoveBody(BaseModel):
    XYZ: str
    DIST: float
    SPD: float | None = None
    ACC: float | None = None
    MODE: str = "relative"
    @field_validator('MODE')
    @classmethod
    def _mode_ok(cls, v):
        v2 = str(v).strip().lower()
        if v2 not in ("relative","absolute"):
            raise ValueError("MODE must be 'relative' or 'absolute'")
        return v2
    @field_validator('XYZ')
    @classmethod
    def _axis_ok(cls, v):
        v2 = str(v).strip().upper()
        ok_vals = {"X","Y","Z","YAW","RZ","P","PITCH","RY"}
        ok = v2 in ok_vals
        if not ok:
            raise ValueError("XYZ must be X/Y/Z/YAW/P")
        return v2
    class Config:
        json_schema_extra = {"example":{"XYZ":"X","DIST":100.0,"SPD":0.5,"ACC":1.0,"MODE":"relative"}}

class StatusBody(BaseModel):
    joint_pos : List[float]
    XYZ_pos   : List[float]
