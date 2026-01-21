# fast_server/app.py
import asyncio, re
from typing import Optional, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fast_server.Model import MoveJBody,AxisMoveBody,StatusBody,PoseMoveBody

def create_app(bridge, cors_origins: Optional[list]=None) -> FastAPI:
    app = FastAPI(title="Robot Unified Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ['*'],
        allow_credentials=True,
        allow_methods=['*'],  # ← 오타 수정
        allow_headers=['*']
    )

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
        print("/move/axis: ",body.AXIS, float(body.DIST), None if body.SPD is None else float(body.SPD), rel)

        bridge.publish_move_axis(body.AXIS, float(body.DIST), None if body.SPD is None else float(body.SPD),
                                 None if body.ACC is None else float(body.ACC), rel)
        return {"ok": True}

    @app.post("/move/vision")
    def move_vision(body: AxisMoveBody):
        rel = (body.MODE.lower()=="relative")
        bridge.publish_move_vision(body.AXIS, float(body.DIST), None if body.SPD is None else float(body.SPD),
                                   None if body.ACC is None else float(body.ACC), rel)
        return {"ok": True}

    @app.post("/move/pose")
    def move_pose(body: PoseMoveBody):
        rel = (body.MODE.lower() == "relative")
        x, y, z, roll, yaw, pitch = [float(v) for v in body.pose]
        bridge.publish_move_pose(x, y, z, roll, yaw, pitch,
                                 None if body.SPD is None else float(body.SPD),
                                 None if body.ACC is None else float(body.ACC),
                                 rel)
        return {"ok": True}

    @app.post("/robot/move_pose")
    def move_pose_robot(body: PoseMoveBody):
        rel = (body.MODE.lower() == "relative")
        x, y, z, roll, yaw, pitch = [float(v) for v in body.pose]
        bridge.publish_move_pose(x, y, z, roll, yaw, pitch,
                                 None if body.SPD is None else float(body.SPD),
                                 None if body.ACC is None else float(body.ACC),
                                 rel)
        return {"ok": True}

    @app.get("/status")
    def get_status():
        # bridge에 캐시가 아직 없더라도 안전하게
        try:
            return bridge.get_latest_status()
        except AttributeError:
            return {}

    return app
