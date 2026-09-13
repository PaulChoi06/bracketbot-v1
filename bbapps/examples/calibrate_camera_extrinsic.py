# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "opencv-python",
#   "fastapi",
#   "uvicorn",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""
Camera -> arm_base extrinsic, measured with the arm itself.

find_camera_extrinsic.py tried to get this for free from the URDF and hit a
dead end: the only nearby link names (head__head, camera_cover__camera_cover)
turned out to be unrelated small parts near the base (near-zero offset from
arm_base, identity rotation, and head__head's own parent joint chain runs
through hub_motor_left__hub_motor_left -- a wheel/base part, not a ~1.5m-high
tilted camera mount). The head/camera assembly simply isn't in this URDF.

So: measure it directly. The arm's forward kinematics (cfg.ik.fk) already
gives an accurate end-effector position in arm_base frame from encoder
readings alone -- no camera needed for that half. Touch the gripper to the
same physical point a detected tag is at, and you have one (camera-frame,
arm-base-frame) correspondence for free. Three or more non-collinear points
and a rigid transform (rotation + translation) is fully determined --
classic hand-eye/Kabsch-style calibration, just done by hand instead of
with a checkerboard rig, which is proportionate to a 4-hour demo budget.

Procedure:
  1. Run this (uv run examples/calibrate_camera_extrinsic.py --arm arm_left),
     open http://<robot>.local:8014/.
  2. With the arm torque OFF (or under teleop), physically move the gripper
     so its fingertip touches a visible tag -- reposition the actual tag
     between samples (move it around the workspace), not just the arm.
  3. When the page shows both a confirmed tag detection AND arm state, hit
     "Capture sample". Repeat at 4+ well-spread, non-collinear points
     (don't cluster them in one corner -- spread across the reachable
     workspace, vary height too).
  4. Hit "Solve & save". Reports RMS fit error (mm) -- if it's much more
     than ~10-15mm, a sample point was probably touched sloppily; reset
     and redo, or drop the worst outlier and re-solve.
  5. Result is written to /home/bracketbot/camera_extrinsic.json as
     {"R": 3x3, "t": [x,y,z]} such that
     p_arm_base = R @ p_camera_frame + t
     -- this is what a later pick-and-place script composes with a tag's
     solvePnP position to get its arm-frame pickup point.

This script only ever READS arm state (Reader("arm_<side>.state")) to run
FK -- it never writes to arm_ctrl or arm_torque. It cannot move the arm.
Positioning the gripper at each sample point is entirely up to you (by
hand, or with an existing teleop script running alongside this one).
"""
import argparse
import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
from bbos import Reader, Config

FPS = 15
MIN_SAMPLES = 3

ARUCO_DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50, "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250, "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100, "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100, "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
PNP_FLAG = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE

app = FastAPI()
STATE = {}
LOCK = threading.Lock()
FRAME_Q = Queue(maxsize=2)
LATEST = {"tag_found": False, "tag_cam_xyz": None, "arm_ready": False, "arm_xyz": None}
SAMPLES = {"cam": [], "arm": []}  # lists of (3,) np arrays, index-aligned
RESULT = {"R": None, "t": None, "rms_mm": None, "saved_to": None}


def crop_eye(stereo, side):
    half = stereo.shape[1] // 2
    return stereo[:, :half] if side == "left" else stereo[:, half:]


def build_detector(dict_name):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)[:2]
    else:
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


def marker_object_points(size):
    s = size / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)


def load_calibration(path, expected_size):
    data = json.loads(Path(path).read_text())
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64).reshape(-1, 1)
    size = tuple(data["image_size"])
    if size != tuple(expected_size):
        print(f"[calibrate_camera_extrinsic] WARNING: calibration image_size {size} != "
              f"live eye size {expected_size}; positions will be off.")
    return K, D


def kabsch(A, B):
    """Rigid transform (R, t) minimizing sum |R@A_i + t - B_i|^2, A,B: (N,3).
    Standard SVD (Kabsch/Procrustes) solution, reflection-corrected."""
    cA, cB = A.mean(axis=0), B.mean(axis=0)
    AA, BB = A - cA, B - cB
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cB - R @ cA
    return R, t


def rms_fit_error_mm(R, t, A, B):
    pred = (R @ A.T).T + t
    return float(np.sqrt(np.mean(np.sum((pred - B) ** 2, axis=1))) * 1000.0)


def arm_state_loop(side):
    cfg = Config(side)
    cfg.ik.init()
    with Reader(f"{side}.state") as r:
        while True:
            if r.ready():
                motor = np.asarray(r.data["pos"], dtype=np.float64).copy()
                urdf8 = cfg.q2urdf(motor)
                pos, _ = cfg.ik.fk(urdf8[:7].tolist())
                with LOCK:
                    LATEST["arm_ready"] = True
                    LATEST["arm_xyz"] = np.array(pos, dtype=np.float64)
            time.sleep(0.02)


def camera_loop(side, calib_path, dict_name, marker_id, marker_size):
    cam = Config("cam_head")
    eye_size = (cam.width // 2, cam.height)
    K, D = load_calibration(calib_path, eye_size)
    detect = build_detector(dict_name)
    obj_pts = marker_object_points(marker_size)

    with Reader("camera.head.jpeg") as r:
        while True:
            if not r.ready():
                time.sleep(0.005)
                continue
            n = int(r.data["jpeg_len"])
            buf = r.data["jpeg"][:n].copy()
            stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if stereo is None:
                continue
            eye = crop_eye(stereo, side)
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            corners, ids = detect(gray)

            found = False
            cam_xyz = None
            annotated = eye.copy()
            if ids is not None:
                ids_flat = ids.flatten().tolist()
                if marker_id in ids_flat:
                    idx = ids_flat.index(marker_id)
                    c = corners[idx]
                    ud = cv2.fisheye.undistortPoints(
                        c.reshape(-1, 1, 2).astype(np.float64), K, D, P=K).reshape(-1, 2)
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, ud, K, None, flags=PNP_FLAG)
                    if ok:
                        found = True
                        cam_xyz = tvec.flatten().astype(np.float64)
                        poly = ud.astype(np.int32)
                        cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
                        cv2.putText(annotated, f"id={marker_id}", tuple(poly[0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            with LOCK:
                LATEST["tag_found"] = found
                LATEST["tag_cam_xyz"] = cam_xyz
                n_samples = len(SAMPLES["cam"])
                arm_ready = LATEST["arm_ready"]

            hud = f"tag={'FOUND' if found else 'none'}  arm={'ready' if arm_ready else 'NO STATE'}  samples={n_samples}"
            cv2.putText(annotated, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if found else (0, 0, 255), 2)

            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                data = jpg.tobytes()
                try:
                    FRAME_Q.put_nowait(data)
                except Exception:
                    try:
                        FRAME_Q.get_nowait()
                        FRAME_Q.put_nowait(data)
                    except Exception:
                        pass


def make_stream():
    async def generate():
        while True:
            try:
                frame = FRAME_Q.get_nowait()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Empty:
                pass
            await asyncio.sleep(1 / FPS)
    return generate


_gen = make_stream()


@app.get("/stream")
async def stream():
    return StreamingResponse(
        _gen(), media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/frame")
async def frame():
    try:
        f = FRAME_Q.get(timeout=0.5)
        return Response(content=f, media_type="image/jpeg")
    except Exception:
        return Response(status_code=503)


@app.get("/status")
async def status():
    with LOCK:
        return {
            "tag_found": LATEST["tag_found"],
            "tag_cam_xyz": LATEST["tag_cam_xyz"].tolist() if LATEST["tag_cam_xyz"] is not None else None,
            "arm_ready": LATEST["arm_ready"],
            "arm_xyz": LATEST["arm_xyz"].tolist() if LATEST["arm_xyz"] is not None else None,
            "samples": len(SAMPLES["cam"]),
            "min_samples": MIN_SAMPLES,
            "last_result": {"rms_mm": RESULT["rms_mm"], "saved_to": RESULT["saved_to"]},
        }


@app.post("/capture")
async def capture():
    with LOCK:
        if not (LATEST["tag_found"] and LATEST["arm_ready"]):
            return {"ok": False, "reason": "need both a confirmed tag and arm state right now"}
        SAMPLES["cam"].append(LATEST["tag_cam_xyz"].copy())
        SAMPLES["arm"].append(LATEST["arm_xyz"].copy())
        return {"ok": True, "samples": len(SAMPLES["cam"])}


@app.post("/reset")
async def reset():
    with LOCK:
        SAMPLES["cam"].clear()
        SAMPLES["arm"].clear()
        RESULT.update(R=None, t=None, rms_mm=None, saved_to=None)
    return {"ok": True}


@app.post("/solve")
async def solve():
    with LOCK:
        A = np.array(SAMPLES["cam"])
        B = np.array(SAMPLES["arm"])
    if len(A) < MIN_SAMPLES:
        return {"ok": False, "reason": f"need >= {MIN_SAMPLES} samples, have {len(A)}"}
    R, t = kabsch(A, B)
    rms_mm = rms_fit_error_mm(R, t, A, B)
    out = {"R": R.tolist(), "t": t.tolist(), "rms_mm": rms_mm,
           "n_samples": len(A), "arm": STATE["arm"], "calibrated_at": time.time()}
    out_path = STATE["out_path"]
    tmp = out_path + ".tmp"
    Path(tmp).write_text(json.dumps(out, indent=2))
    Path(tmp).replace(out_path)
    with LOCK:
        RESULT.update(R=R, t=t, rms_mm=rms_mm, saved_to=out_path)
    return {"ok": True, **out}


@app.get("/")
async def index():
    html = f"""
    <html><head><title>Camera Extrinsic Calibration</title>
    <style>
      body {{ background:#000; color:#eee; font-family:sans-serif; margin:0; padding:20px; }}
      img {{ max-width:100%; display:block; margin-bottom:14px; border:1px solid #333; }}
      button {{ padding:8px 16px; margin:0 8px 12px 0; font-size:14px; cursor:pointer; }}
      #status {{ white-space:pre; background:#111; padding:12px; border-radius:6px; max-width:600px; }}
    </style></head>
    <body>
      <h2>Camera -> {STATE['arm']} extrinsic calibration</h2>
      <p>Touch the gripper to tag id={STATE['marker_id']} at a spread of points across the
         workspace (move the tag between samples), capturing at each. Need >= {MIN_SAMPLES},
         more and more spread out is better.</p>
      <img src="/stream">
      <div>
        <button onclick="post('/capture')">Capture sample</button>
        <button onclick="post('/solve')">Solve &amp; save</button>
        <button onclick="post('/reset')">Reset</button>
      </div>
      <pre id="status">loading...</pre>
      <script>
        async function post(path) {{
          const r = await fetch(path, {{method: 'POST'}});
          document.getElementById('status').textContent = JSON.stringify(await r.json(), null, 2);
        }}
        async function refresh() {{
          try {{
            const r = await fetch('/status');
            document.getElementById('status').textContent = JSON.stringify(await r.json(), null, 2);
          }} catch (e) {{}}
        }}
        setInterval(refresh, 500);
        refresh();
      </script>
    </body></html>
    """
    return Response(content=html, media_type="text/html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], default="left",
                     help="which eye of the head stereo camera to use")
    ap.add_argument("--arm", choices=["arm_left", "arm_right"], default="arm_left")
    ap.add_argument("--calib", type=str, default=None,
                     help="fisheye calibration json (default /home/bracketbot/fisheye_calib_<side>.json)")
    ap.add_argument("--dict", dest="dict_name", type=str, default="4X4_50", choices=list(ARUCO_DICTS))
    ap.add_argument("--marker-id", type=int, required=True,
                     help="the specific tag id you'll be touching with the gripper")
    ap.add_argument("--marker-size", type=float, default=0.069,
                     help="marker side length in meters (default 0.069 = 69mm)")
    ap.add_argument("--out", type=str, default="/home/bracketbot/camera_extrinsic.json")
    ap.add_argument("--port", type=int, default=8014)
    args = ap.parse_args()

    calib_path = args.calib or f"/home/bracketbot/fisheye_calib_{args.side}.json"
    if not Path(calib_path).exists():
        raise SystemExit(
            f"No fisheye calibration at {calib_path}.\n"
            f"Run: uv run examples/calibrate_fisheye.py --side {args.side}\n"
        )

    STATE.update(arm=args.arm, marker_id=args.marker_id, out_path=args.out)

    threading.Thread(target=arm_state_loop, args=(args.arm,), daemon=True).start()
    threading.Thread(target=camera_loop, args=(args.side, calib_path, args.dict_name,
                                                args.marker_id, args.marker_size), daemon=True).start()

    host = socket.gethostname()
    print(f"[+] Camera extrinsic calibration on http://{host}.local:{args.port}/")
    print(f"[+] Reading arm state from {args.arm}.state (read-only, cannot move the arm)")
    print(f"[+] Touch tag id={args.marker_id} with the gripper at >= {MIN_SAMPLES} spread points")
    print(f"[+] Output -> {args.out}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error",
                access_log=False, timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
