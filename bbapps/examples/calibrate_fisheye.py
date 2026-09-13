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
Fisheye intrinsic calibration for one eye of the head stereo camera.

Reads camera.head.jpeg (the combined stereo frame), crops the chosen eye,
and runs the standard cv2.fisheye calibration against a live-detected
calibration target. Serves a live annotated MJPEG feed plus a small control
page (same pattern as view_camera.py) so this can be driven from a browser
while someone else holds the target in front of the robot.

Two target modes (--target):
  checkerboard (default) -- a printed chessboard. ~50 corner correspondences
    per frame, so it converges from relatively few (MIN_SAMPLES_CHECKERBOARD)
    varied views. Recommended when you have a board handy.
  aruco -- a single ArUco marker (e.g. the same tag view_pnp.py tracks), 4
    corners per frame. Fewer points/frame means the fit is noisier and needs
    many more views (MIN_SAMPLES_ARUCO) with deliberately wide coverage --
    push it right into the corners/edges of the frame, not just center and
    fronto-parallel -- to constrain the fisheye distortion terms well,
    especially at the periphery where a small marker held at a normal
    distance never reaches.

Run:    uv run examples/calibrate_fisheye.py --side left
        uv run examples/calibrate_fisheye.py --side left --target aruco --dict 4X4_50 --marker-id 0 --marker-size 0.069
Open:   http://<robot>.local:8010/

Frames are captured automatically whenever the target is detected (subject
to AUTO_CAPTURE_COOLDOWN_S so the same pose isn't captured many times); the
web page also has a "Capture now" button and an auto-capture toggle. Once
you have enough varied views, hit "Run calibration". The result (K, D, image
size) is written to --out (default /home/bracketbot/fisheye_calib_<side>.json)
-- view_pnp.py reads that file.
"""
import argparse
import asyncio
import json
import re
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
from bbos import Reader

MIN_SAMPLES_CHECKERBOARD = 12
# 4 points/frame vs the checkerboard's ~50 -- needs far more views to reach
# comparable constraint on the distortion terms.
MIN_SAMPLES_ARUCO = 30
AUTO_CAPTURE_COOLDOWN_S = 1.5
FPS = 15

ARUCO_DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50, "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250, "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100, "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100, "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

app = FastAPI()
STATE = {"side": "left", "target": "checkerboard", "cols": 9, "rows": 6, "square_size": 0.025,
         "dict": "4X4_50", "marker_id": 0, "marker_size": 0.069, "min_motion_px": 20.0,
         "min_samples": MIN_SAMPLES_CHECKERBOARD, "out_path": None, "auto": True}
LOCK = threading.Lock()
FRAME_Q = Queue(maxsize=2)
SAMPLES = {"obj": [], "img": []}
LATEST = {"found": False, "objp": None, "imgp": None, "draw_corners": None,
          "gray_shape": None, "last_capture_t": 0.0, "last_capture_imgp": None}
RESULT = {"rms": None, "n": 0, "dropped": [], "saved_to": None}


def crop_eye(stereo, side):
    half = stereo.shape[1] // 2
    return stereo[:, :half] if side == "left" else stereo[:, half:]


def build_aruco_detector(dict_name):
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


def find_checkerboard(gray, cols, rows, square_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if not found:
        return False, None, None, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
    objp = np.zeros((1, cols * rows, 3), np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[0, :, :2] = grid * square_size
    imgp = corners.reshape(1, -1, 2).astype(np.float64)
    return True, corners, objp, imgp


def find_marker(gray, detector, marker_id, marker_size):
    """4 corners of one ArUco marker, ordered (TL, TR, BR, BL) to match what
    cv2.aruco returns -- same convention view_pnp.py uses for PnP."""
    corners, ids = detector(gray)
    if ids is None:
        return False, None, None, None
    ids_flat = ids.flatten().tolist()
    if marker_id >= 0:
        if marker_id not in ids_flat:
            return False, None, None, None
        idx = ids_flat.index(marker_id)
    else:
        idx = 0  # any marker is fine when no specific id is requested
    c = corners[idx]  # shape (1, 4, 2)
    s = marker_size / 2.0
    objp = np.array([[[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]], dtype=np.float64)
    imgp = c.reshape(1, 4, 2).astype(np.float64)
    return True, c, objp, imgp


def mean_corner_shift(imgp_a, imgp_b):
    """Mean per-point pixel distance between two (1, N, 2) point sets, as a
    cheap pixel-space stand-in for "did the target actually move"."""
    a, b = imgp_a.reshape(-1, 2), imgp_b.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _add_sample_locked(objp, imgp):
    """Caller must hold LOCK."""
    SAMPLES["obj"].append(objp)
    SAMPLES["img"].append(imgp)
    LATEST["last_capture_t"] = time.monotonic()
    LATEST["last_capture_imgp"] = imgp


def run_fisheye_calibration(objpoints, imgpoints, image_size, max_drops=5):
    """cv2.fisheye.calibrate with CALIB_CHECK_COND raises cv2.error naming the
    index of an ill-conditioned view -- drop it and retry rather than making
    the operator hunt for which of N captures was bad.

    Separately, a set of views with too little geometric variation between
    them (e.g. a target that was held nearly still while auto-capture ran)
    makes cv2.fisheye.calibrate fail its *initial* estimate with an opaque,
    unrelated-looking native assertion (seen in practice: a matrix.cpp
    'rowRange' assertion) rather than the CALIB_CHECK_COND error above.
    Recognize that case too and say what it actually means -- it is not a
    sample-count problem, adding more near-identical views won't fix it."""
    obj, img, dropped = list(objpoints), list(imgpoints), []
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
             | cv2.fisheye.CALIB_CHECK_COND
             | cv2.fisheye.CALIB_FIX_SKEW)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    for _ in range(max_drops + 1):
        n = len(obj)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3)) for _ in range(n)]
        tvecs = [np.zeros((1, 1, 3)) for _ in range(n)]
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj, img, image_size, K, D, rvecs, tvecs, flags, criteria)
            return float(rms), K, D, dropped
        except cv2.error as e:
            msg = str(e)
            m = re.search(r"input array (\d+)", msg)
            if m and len(obj) > 6:
                idx = int(m.group(1))
                dropped.append(idx)
                obj.pop(idx)
                img.pop(idx)
                continue
            raise RuntimeError(
                "OpenCV could not find a valid initial calibration from these views. "
                "This is almost always caused by too little variation between captures "
                "(target held still, or only ever centered/fronto-parallel) -- it is not "
                "fixed by collecting more samples in the same spot. Reset and recapture "
                f"with the target moved to clearly different positions/distances/tilts. "
                f"(raw OpenCV error: {msg})"
            ) from e
    raise RuntimeError("calibration kept failing even after dropping ill-conditioned views")


def camera_loop():
    target = STATE["target"]
    cols, rows, square = STATE["cols"], STATE["rows"], STATE["square_size"]
    detector = build_aruco_detector(STATE["dict"]) if target == "aruco" else None
    marker_id, marker_size = STATE["marker_id"], STATE["marker_size"]

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
            eye = crop_eye(stereo, STATE["side"])
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            if target == "checkerboard":
                found, draw_corners, objp, imgp = find_checkerboard(gray, cols, rows, square)
            else:
                found, draw_corners, objp, imgp = find_marker(gray, detector, marker_id, marker_size)

            annotated = eye.copy()
            if found:
                if target == "checkerboard":
                    cv2.drawChessboardCorners(annotated, (cols, rows), draw_corners, found)
                else:
                    cv2.polylines(annotated, [draw_corners.reshape(-1, 1, 2).astype(np.int32)],
                                  True, (0, 255, 0), 2)

            moved_enough = True
            with LOCK:
                LATEST["found"] = found
                LATEST["objp"] = objp
                LATEST["imgp"] = imgp
                LATEST["gray_shape"] = (int(gray.shape[1]), int(gray.shape[0]))  # (w, h)
                if found:
                    last_imgp = LATEST["last_capture_imgp"]
                    moved_enough = (last_imgp is None
                                     or mean_corner_shift(imgp, last_imgp) >= STATE["min_motion_px"])
                    if STATE["auto"] and moved_enough and \
                            time.monotonic() - LATEST["last_capture_t"] > AUTO_CAPTURE_COOLDOWN_S:
                        _add_sample_locked(objp, imgp)
                n_samples = len(SAMPLES["obj"])

            hud = f"samples={n_samples}  target={'FOUND' if found else 'none'}  side={STATE['side']}"
            if RESULT["rms"] is not None:
                hud += f"  last_rms={RESULT['rms']:.3f}px"
            cv2.putText(annotated, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            if found and not moved_enough:
                cv2.putText(annotated, "move target more before next capture", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            if not STATE["auto"]:
                cv2.putText(annotated, "auto-capture OFF", (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 200, 255), 2)

            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
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
        s = {
            "side": STATE["side"], "target": STATE["target"], "target_found": LATEST["found"],
            "samples": len(SAMPLES["obj"]), "min_samples": STATE["min_samples"],
            "min_motion_px": STATE["min_motion_px"],
            "auto_capture": STATE["auto"], "last_rms_px": RESULT["rms"],
            "dropped_on_last_calibration": RESULT["dropped"], "saved_to": RESULT["saved_to"],
        }
        if STATE["target"] == "checkerboard":
            s["board_cols"] = STATE["cols"]; s["board_rows"] = STATE["rows"]
            s["square_size_m"] = STATE["square_size"]
        else:
            s["dict"] = STATE["dict"]; s["marker_id"] = STATE["marker_id"]
            s["marker_size_m"] = STATE["marker_size"]
        return s


@app.post("/capture")
async def capture():
    with LOCK:
        if not LATEST["found"]:
            return {"ok": False, "reason": "no target in view right now"}
        _add_sample_locked(LATEST["objp"], LATEST["imgp"])
        return {"ok": True, "samples": len(SAMPLES["obj"])}


@app.post("/reset")
async def reset():
    with LOCK:
        SAMPLES["obj"].clear()
        SAMPLES["img"].clear()
        LATEST["last_capture_imgp"] = None
        RESULT.update(rms=None, n=0, dropped=[], saved_to=None)
    return {"ok": True}


@app.post("/auto/{enabled}")
async def set_auto(enabled: str):
    STATE["auto"] = enabled.lower() in ("1", "true", "on", "yes")
    return {"auto_capture": STATE["auto"]}


@app.post("/calibrate")
async def calibrate():
    with LOCK:
        obj = list(SAMPLES["obj"])
        img = list(SAMPLES["img"])
        gshape = LATEST["gray_shape"]
    min_samples = STATE["min_samples"]
    if len(obj) < min_samples:
        return {"ok": False, "reason": f"need >= {min_samples} samples, have {len(obj)}"}
    if gshape is None:
        return {"ok": False, "reason": "no frame seen yet"}
    try:
        rms, K, D, dropped = run_fisheye_calibration(obj, img, gshape)
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    target_info = ({"cols": STATE["cols"], "rows": STATE["rows"], "square_size_m": STATE["square_size"]}
                    if STATE["target"] == "checkerboard" else
                    {"dict": STATE["dict"], "marker_id": STATE["marker_id"], "marker_size_m": STATE["marker_size"]})
    out = {
        "side": STATE["side"], "image_size": [gshape[0], gshape[1]],
        "K": K.tolist(), "D": D.reshape(-1).tolist(), "rms_px": rms,
        "n_samples": len(obj) - len(dropped),
        "target": STATE["target"], "target_info": target_info,
        "calibrated_at": time.time(),
    }
    # Write-then-rename: a write failure (disk, permissions, ...) can never
    # leave a truncated/corrupt file in place of a previously good one.
    out_path = Path(STATE["out_path"])
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(out, indent=2))
    tmp_path.replace(out_path)
    with LOCK:
        RESULT.update(rms=rms, n=out["n_samples"], dropped=dropped, saved_to=STATE["out_path"])
    return {"ok": True, **out}


@app.get("/")
async def index():
    if STATE["target"] == "checkerboard":
        target_desc = (f"Checkerboard: {STATE['cols']}x{STATE['rows']} inner corners, "
                        f"{STATE['square_size']*1000:.1f} mm squares.")
    else:
        target_desc = (f"ArUco marker: dict={STATE['dict']} id={STATE['marker_id']} "
                        f"size={STATE['marker_size']*1000:.0f}mm. Only 4 points/frame -- "
                        f"needs {STATE['min_samples']}+ views with wide coverage (push it into "
                        f"the corners/edges of the frame, not just center).")
    html = f"""
    <html><head><title>Fisheye Calibration ({STATE['side']})</title>
    <style>
      body {{ background:#000; color:#eee; font-family:sans-serif; margin:0; padding:20px; }}
      img {{ max-width:100%; display:block; margin-bottom:14px; border:1px solid #333; }}
      button {{ padding:8px 16px; margin:0 8px 12px 0; font-size:14px; cursor:pointer; }}
      #status {{ white-space:pre; background:#111; padding:12px; border-radius:6px; max-width:600px; }}
    </style></head>
    <body>
      <h2>Fisheye calibration &mdash; {STATE['side']} eye</h2>
      <p>{target_desc} Move it through edges/corners/tilts of the frame &mdash;
         auto-capture skips frames that haven't moved &ge;{STATE['min_motion_px']:.0f}px
         from the last capture, so holding it still won't pile up useless duplicate views.</p>
      <img src="/stream">
      <div>
        <button onclick="post('/capture')">Capture now</button>
        <button onclick="post('/calibrate')">Run calibration</button>
        <button onclick="post('/reset')">Reset samples</button>
        <button onclick="toggleAuto()">Toggle auto-capture</button>
      </div>
      <pre id="status">loading...</pre>
      <script>
        async function post(path) {{
          const r = await fetch(path, {{method: 'POST'}});
          document.getElementById('status').textContent = JSON.stringify(await r.json(), null, 2);
        }}
        async function toggleAuto() {{
          const cur = await (await fetch('/status')).json();
          await post(`/auto/${{!cur.auto_capture}}`);
        }}
        async function refresh() {{
          try {{
            const r = await fetch('/status');
            document.getElementById('status').textContent = JSON.stringify(await r.json(), null, 2);
          }} catch (e) {{}}
        }}
        setInterval(refresh, 1000);
        refresh();
      </script>
    </body></html>
    """
    return Response(content=html, media_type="text/html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], default="left",
                     help="which eye of the head stereo camera to calibrate")
    ap.add_argument("--target", choices=["checkerboard", "aruco"], default="checkerboard",
                     help="calibration target type (default checkerboard)")
    ap.add_argument("--cols", type=int, default=9, help="[checkerboard] inner corners, horizontal")
    ap.add_argument("--rows", type=int, default=6, help="[checkerboard] inner corners, vertical")
    ap.add_argument("--square-size", type=float, default=0.025,
                     help="[checkerboard] square size in meters (default 0.025 = 25mm)")
    ap.add_argument("--dict", dest="dict_name", type=str, default="4X4_50", choices=list(ARUCO_DICTS),
                     help="[aruco] marker dictionary")
    ap.add_argument("--marker-id", type=int, default=0,
                     help="[aruco] marker id to use; -1 accepts any detected marker")
    ap.add_argument("--marker-size", type=float, default=0.069,
                     help="[aruco] marker side length in meters, black border included (default 0.069 = 69mm)")
    ap.add_argument("--min-samples", type=int, default=None,
                     help="override the minimum views required before /calibrate will run")
    ap.add_argument("--min-motion-px", type=float, default=20.0,
                     help="auto-capture skips a detection whose corners moved less than this "
                          "many pixels from the last captured sample (default 20); prevents "
                          "a still target from piling up near-duplicate, degenerate views")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--out", type=str, default=None,
                     help="output calibration json path (default /home/bracketbot/fisheye_calib_<side>.json)")
    args = ap.parse_args()

    STATE["side"] = args.side
    STATE["target"] = args.target
    STATE["cols"] = args.cols
    STATE["rows"] = args.rows
    STATE["square_size"] = args.square_size
    STATE["dict"] = args.dict_name
    STATE["marker_id"] = args.marker_id
    STATE["marker_size"] = args.marker_size
    default_min = MIN_SAMPLES_CHECKERBOARD if args.target == "checkerboard" else MIN_SAMPLES_ARUCO
    STATE["min_samples"] = args.min_samples or default_min
    STATE["min_motion_px"] = args.min_motion_px
    STATE["out_path"] = args.out or f"/home/bracketbot/fisheye_calib_{args.side}.json"

    threading.Thread(target=camera_loop, daemon=True).start()

    host = socket.gethostname()
    print(f"[+] Fisheye calibration ({args.side} eye) on http://{host}.local:{args.port}/")
    if args.target == "checkerboard":
        print(f"[+] Target: checkerboard {args.cols}x{args.rows} inner corners, "
              f"{args.square_size*1000:.1f} mm squares (min {STATE['min_samples']} samples)")
    else:
        print(f"[+] Target: ArUco dict={args.dict_name} id={args.marker_id} "
              f"size={args.marker_size*1000:.0f}mm (min {STATE['min_samples']} samples)")
    print(f"[+] Output -> {STATE['out_path']}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error",
                access_log=False, timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
