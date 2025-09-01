# fast_server/app.py
import asyncio, re
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

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

def create_app(bridge, cors_origins: Optional[list]=None) -> FastAPI:
    app = FastAPI(title="Robot Unified Server")
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins or ['*'],
                       allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

    @app.get("/mjpeg")
    async def mjpeg():
        boundary = b'--frame'
        headers = {"Cache-Control":"no-store, no-cache, must-revalidate, max-age=0",
                   "Pragma":"no-cache", "Expires":"0", "Connection":"close", "X-Accel-Buffering":"no"}
        async def gen():
            SLEEP=0.001; first=True
            while True:
                frame = bridge.get_latest_jpeg()
                if frame is None:
                    await asyncio.sleep(SLEEP); continue
                prefix = boundary if first else b'\r\n'+boundary; first=False
                part_headers = b'\r\n'.join([b'Content-Type: image/jpeg', f'Content-Length: {len(frame)}'.encode(), b'', b''])
                yield prefix + b'\r\n' + part_headers + frame
                await asyncio.sleep(SLEEP)
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame', headers=headers)

    @app.get("/detections")
    def detections():
        return bridge.get_latest_detections()

    @app.post("/robot/movej")
    def movej(body: MoveJBody):
        bridge.publish_movej(body.joints, body.speed, body.accel, body.relative)
        return {"ok": True}

    @app.post("/move/axis")
    def move_axis(body: AxisMoveBody):
        rel = (body.MODE.lower()=="relative")
        bridge.publish_move_axis(body.AXIS, float(body.DIST), None if body.SPD is None else float(body.SPD),
                                 None if body.ACC is None else float(body.ACC), rel)
        return {"ok": True}

    @app.post("/move/XYZ")
    def move_xyz(body: AxisMoveBody):
        axis = body.AXIS.upper()
        if axis not in ("X","Y","Z"):
            raise HTTPException(status_code=400, detail="AXIS must be X/Y/Z for /move/XYZ")
        rel = (body.MODE.lower()=="relative")
        bridge.publish_move_xyz(axis, float(body.DIST), None if body.SPD is None else float(body.SPD),
                                None if body.ACC is None else float(body.ACC), rel)
        return {"ok": True}

    @app.post("/move/vision")
    def move_vision(body: AxisMoveBody):
        rel = (body.MODE.lower()=="relative")
        bridge.publish_move_vision(body.AXIS, float(body.DIST), None if body.SPD is None else float(body.SPD),
                                   None if body.ACC is None else float(body.ACC), rel)
        return {"ok": True}

    return app
