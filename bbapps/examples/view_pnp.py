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
Live ArUco pose (PnP) viewer for one eye of the head stereo camera.

Detects ArUco markers on camera.head.jpeg, undistorts their corners using a
fisheye calibration (from calibrate_fisheye.py), solves PnP per marker, and
streams an annotated MJPEG feed -- same web-feed pattern as view_camera.py --
with marker outline, pose axes, and a live pose readout (position, distance,
orientation).

Run:    uv run examples/view_pnp.py --side left
Open:   http://<robot>.local:8011/

Requires a calibration file from calibrate_fisheye.py for the SAME --side
(default /home/bracketbot/fisheye_calib_<side>.json). Run that first:
    uv run examples/calibrate_fisheye.py --side left

Defaults match a 4x4_50 dictionary, marker id 0, 69mm marker size -- override
with --dict / --marker-id / --marker-size for a different tag. --marker-id
now defaults to -1 (treat any decoded id as the target) since at range the
decoded id itself becomes unreliable -- see below.

Detection range: a far marker only occupies a handful of pixels, and
cv2.aruco's default detector params (tuned to reject noise) discard
candidates that small. Knobs, in order of effect:
  --error-correction-rate  maxed out (1.0) by default. This is the one that
                         matters most once id correctness stops mattering:
                         a barely-legible marker's bit pattern normally gets
                         REJECTED outright as "not a valid code"; maxing this
                         tells the decoder to accept it as the closest valid
                         id anyway rather than discarding the detection. The
                         reported id at extreme range is then essentially
                         arbitrary -- fine, since specificity isn't needed --
                         but corner order/orientation still comes from the
                         real decode step, unlike a pure shape-only guess.
  --min-perimeter-rate  lowers the minimum marker size (as a fraction of
                         image size) the detector will even consider.
  --upscale-steps       on frames where native-resolution detection finds
                         nothing, retries on progressively larger upscaled
                         copies (more pixels on target = past the detector's
                         internal thresholds) until one hits, then maps the
                         corners back down. Costs more CPU the farther it has
                         to climb the ladder.
Also widened: the adaptive-threshold window search (more scales tried) and
the tolerance for a noisy/blurred black border. All of this trades false
positives and CPU for range -- if it starts latching onto things that
aren't the marker, dial --error-correction-rate and --min-perimeter-rate
back down. Pose/orientation accuracy at range stays inherently noisy (few
pixels = noisy corners); these knobs are about *noticing it's there*
sooner, not about accurate pose from far away.
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

FPS = 20

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
LATEST_POSES = []


def crop_eye(stereo, side):
    half = stereo.shape[1] // 2
    return stereo[:, :half] if side == "left" else stereo[:, half:]


def build_detector(dict_name, min_perimeter_rate=0.008, error_correction_rate=0.8):
    """Tuned for range over precision, but NOT maxed out -- error_correction_rate
    at its ceiling (1.0) turned out to make almost any small textured quad in
    the scene (chair casters, clothing wrinkles, cable loops) decode as some
    valid id; that false-positive flood is worse than the extra range is
    worth. This is a middle ground; camera_loop's temporal confirmation
    (a candidate must repeat near the same spot for CONFIRM_HITS frames) is
    what actually makes it safe to lean permissive here at all.
      minMarkerPerimeterRate  cv2's own default (0.03) requires a marker's
        perimeter to be >=3% of the image's longer side -- a distant 69mm
        tag never reaches that. 0.008 is a middle ground (free, just a
        laxer accept threshold on candidate quads).
      errorCorrectionRate  the id-decode step: normally a marker whose bits
        can't be read cleanly is REJECTED outright as "not a valid code".
        0.8 (cv2's own default is 0.6) accepts more borderline reads as the
        closest valid id -- reported id gets less reliable, acceptable since
        specificity isn't needed -- without going as far as 1.0, which
        accepted almost anything.
      maxErroneousBitsInBorderRate  mildly relaxed (cv2 default 0.35) to
        tolerate a blurred/noisy black border, the first (pre-id) check that
        also kills small markers.
      adaptiveThreshWinSize{Min,Max,Step}  widened/finer search over
        binarization window sizes, since a tiny marker needs a different
        threshold scale than the rest of the scene. This doesn't loosen
        what counts as a match, just searches harder for one -- costs more
        CPU, not more false positives.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])

    def configure(params):
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.minMarkerPerimeterRate = min_perimeter_rate
        params.errorCorrectionRate = error_correction_rate
        if hasattr(params, "maxErroneousBitsInBorderRate"):
            params.maxErroneousBitsInBorderRate = 0.45
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 40
        params.adaptiveThreshWinSizeStep = 4
        return params

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = configure(cv2.aruco.DetectorParameters())
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)[:2]
    else:
        params = configure(cv2.aruco.DetectorParameters_create())
        return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


def detect_with_upscale(detect, gray, steps):
    """Try native resolution first (cheap, handles the common close-range
    case); only pay for upscaled retries on frames where that finds nothing,
    climbing `steps` (ascending factors) until one hits or they're exhausted.
    Returns (corners, ids, scale_used) -- corners are still in the scale
    they were detected at, caller must divide by scale_used."""
    corners, ids = detect(gray)
    if ids is not None:
        return corners, ids, 1.0
    scale = 1.0
    for s in steps:
        if s <= 1.0:
            continue
        scale = s
        big = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        corners, ids = detect(big)
        if ids is not None:
            return corners, ids, s
    return corners, ids, scale


def marker_object_points(size):
    """Corners in marker frame, ordered (TL, TR, BR, BL) to match cv2.aruco's
    detected-corner order -- the convention cv2.aruco.estimatePoseSingleMarkers
    itself uses internally."""
    s = size / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)


def rot_to_euler_deg(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def load_calibration(path, expected_size):
    data = json.loads(Path(path).read_text())
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64).reshape(-1, 1)
    size = tuple(data["image_size"])
    if size != tuple(expected_size):
        print(f"[view_pnp] WARNING: calibration image_size {size} != live eye size "
              f"{expected_size}; poses will be wrong. Recalibrate at this resolution.")
    return K, D


DEFAULT_CONFIRM_HITS = 2  # consecutive near-same-spot detections before a candidate is trusted
TRACK_MATCH_PX = 60       # undistorted-pixel radius within which two detections count as "the same spot"
TRACK_MAX_AGE_S = 1.0     # drop a track that hasn't been re-seen this long


def update_tracks(tracks, now, det_id, center):
    """Match (id, pixel center) against recent tracks; same id within
    TRACK_MATCH_PX counts as a repeat sighting, anything else starts a new
    track. A one-off false read (chair caster, cable loop, ...) essentially
    never lands on the same id+spot twice in a row, so it never accumulates
    enough hits; a real marker -- even a jittery far one -- does. Returns the
    matched/created track dict."""
    for t in tracks:
        if t["id"] == det_id and np.linalg.norm(t["center"] - center) < TRACK_MATCH_PX:
            t["center"] = center
            t["hits"] += 1
            t["t"] = now
            return t
    t = {"id": det_id, "center": center, "hits": 1, "t": now}
    tracks.append(t)
    return t


def camera_loop(side, K, D, map1, map2, dict_name, marker_id, marker_size,
                 min_perimeter_rate, error_correction_rate, upscale_steps,
                 max_distance_m, confirm_hits):
    detect = build_detector(dict_name, min_perimeter_rate, error_correction_rate)
    obj_pts = marker_object_points(marker_size)
    max_dist_mm = max_distance_m * 1000.0
    tracks = []

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
            corners, ids, scale = detect_with_upscale(detect, gray, upscale_steps)
            undistorted = cv2.remap(eye, map1, map2, interpolation=cv2.INTER_LINEAR)

            now = time.monotonic()
            tracks[:] = [t for t in tracks if now - t["t"] < TRACK_MAX_AGE_S]

            poses = []
            unconfirmed = 0
            if ids is not None:
                for c, i in zip(corners, ids.flatten()):
                    c_native = (c / scale) if scale != 1.0 else c
                    ud = cv2.fisheye.undistortPoints(
                        c_native.reshape(-1, 1, 2).astype(np.float64), K, D, P=K).reshape(-1, 2)
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, ud, K, None, flags=PNP_FLAG)
                    if not ok:
                        continue
                    dist_mm = float(np.linalg.norm(tvec) * 1000)
                    if dist_mm > max_dist_mm:
                        continue  # sanity cutoff -- room isn't that big, this is noise

                    track = update_tracks(tracks, now, int(i), ud.mean(axis=0))
                    poly = ud.astype(np.int32)
                    if track["hits"] < confirm_hits:
                        unconfirmed += 1
                        cv2.polylines(undistorted, [poly], True, (90, 90, 90), 1)
                        continue  # not trusted yet -- drawn faint, not reported

                    R, _ = cv2.Rodrigues(rvec)
                    is_target = (marker_id < 0) or (int(i) == marker_id)
                    color = (0, 255, 0) if is_target else (0, 165, 255)
                    cv2.polylines(undistorted, [poly], True, color, 2)
                    cv2.drawFrameAxes(undistorted, K, None, rvec, tvec, marker_size * 0.75)
                    cv2.putText(undistorted, f"id={int(i)} d={dist_mm:.0f}mm",
                                (int(poly[0][0]), max(0, int(poly[0][1]) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    poses.append({
                        "id": int(i),
                        "tvec_mm": [round(v, 2) for v in (tvec.flatten() * 1000).tolist()],
                        "euler_deg": [round(v, 2) for v in rot_to_euler_deg(R).tolist()],
                        "distance_mm": round(dist_mm, 2),
                    })

            hud = f"side={side} target_id={marker_id} markers={len(poses)}"
            if unconfirmed:
                hud += f"  unconfirmed={unconfirmed}"
            if scale != 1.0:
                hud += f"  upscaled={scale:.1f}x"
            cv2.putText(undistorted, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            with LOCK:
                LATEST_POSES[:] = poses

            ok, jpg = cv2.imencode(".jpg", undistorted, [cv2.IMWRITE_JPEG_QUALITY, 85])
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


@app.get("/pose")
async def pose():
    with LOCK:
        return {"markers": list(LATEST_POSES)}


@app.get("/")
async def index():
    html = f"""
    <html><head><title>ArUco PnP ({STATE.get('side')})</title>
    <style>
      body {{ background:#000; color:#eee; font-family:sans-serif; margin:0; padding:20px; }}
      img {{ max-width:100%; display:block; margin-bottom:14px; border:1px solid #333; }}
      #poses {{ white-space:pre; background:#111; padding:12px; border-radius:6px; max-width:600px; }}
    </style></head>
    <body>
      <h2>ArUco PnP &mdash; {STATE.get('side')} eye</h2>
      <p>dict={STATE.get('dict')} target_id={STATE.get('marker_id')} marker_size={STATE.get('marker_size', 0)*1000:.0f}mm</p>
      <img src="/stream">
      <pre id="poses">loading...</pre>
      <script>
        async function refresh() {{
          try {{
            const r = await fetch('/pose');
            const j = await r.json();
            document.getElementById('poses').textContent =
              j.markers.length ? JSON.stringify(j.markers, null, 2) : "(no markers detected)";
          }} catch (e) {{}}
        }}
        setInterval(refresh, 300);
        refresh();
      </script>
    </body></html>
    """
    return Response(content=html, media_type="text/html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], default="left",
                     help="which eye of the head stereo camera to use")
    ap.add_argument("--calib", type=str, default=None,
                     help="calibration json path (default /home/bracketbot/fisheye_calib_<side>.json)")
    ap.add_argument("--dict", dest="dict_name", type=str, default="4X4_50", choices=list(ARUCO_DICTS))
    ap.add_argument("--marker-id", type=int, default=-1,
                     help="marker id to highlight as the target; -1 (default) treats any "
                          "detected id as the target, since decoded id becomes unreliable at "
                          "the ranges --error-correction-rate is pushing for")
    ap.add_argument("--marker-size", type=float, default=0.069,
                     help="marker side length in meters (default 0.069 = 69mm)")
    ap.add_argument("--min-perimeter-rate", type=float, default=0.008,
                     help="minimum marker perimeter as a fraction of image size (default 0.008; "
                          "cv2's own default is 0.03 -- lower catches smaller/farther markers, "
                          "free of extra compute)")
    ap.add_argument("--error-correction-rate", type=float, default=0.8,
                     help="bit-error tolerance when decoding a marker's id, 0-1 (default 0.8; "
                          "cv2's own default is 0.6, max is 1.0). Higher accepts more borderline "
                          "reads as the closest valid id instead of discarding them -- reported "
                          "id gets less reliable, fine when specificity doesn't matter, but "
                          "pushing this too high (near 1.0) makes near-anything textured in the "
                          "scene register as 'a marker' -- --confirm-hits is what keeps that safe")
    ap.add_argument("--upscale-steps", type=str, default="2,3.5,5",
                     help="comma-separated upscale factors to try in order, on frames where "
                          "native-res detection finds nothing, until one detects (default "
                          "2,3.5,5; each step costs more CPU, only paid when needed)")
    ap.add_argument("--confirm-hits", type=int, default=DEFAULT_CONFIRM_HITS,
                     help=f"consecutive near-same-spot detections required before a candidate "
                          f"is trusted and shown/reported (default {DEFAULT_CONFIRM_HITS}); the "
                          f"actual defense against false positives from permissive detector "
                          f"settings -- raise it if junk is still getting confirmed, lower it "
                          f"(to 1) to see every raw candidate, confirmed or not")
    ap.add_argument("--max-distance-m", type=float, default=8.0,
                     help="drop any solved pose farther than this (default 8.0m) as an obvious "
                          "false positive -- adjust to roughly your room's size")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    upscale_steps = sorted(float(x) for x in args.upscale_steps.split(",") if x.strip())

    calib_path = args.calib or f"/home/bracketbot/fisheye_calib_{args.side}.json"
    if not Path(calib_path).exists():
        raise SystemExit(
            f"No calibration file at {calib_path}.\n"
            f"Run this first:  uv run examples/calibrate_fisheye.py --side {args.side}\n"
            f"...then click \"Run calibration\" on its web page, then re-run view_pnp.py."
        )

    cam = Config("cam_head")
    eye_size = (cam.width // 2, cam.height)
    K, D = load_calibration(calib_path, eye_size)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, eye_size, cv2.CV_16SC2)

    STATE.update(side=args.side, dict=args.dict_name, marker_id=args.marker_id,
                 marker_size=args.marker_size, calib_path=calib_path)

    threading.Thread(target=camera_loop, args=(args.side, K, D, map1, map2, args.dict_name,
                                                args.marker_id, args.marker_size,
                                                args.min_perimeter_rate, args.error_correction_rate,
                                                upscale_steps, args.max_distance_m, args.confirm_hits),
                     daemon=True).start()

    host = socket.gethostname()
    print(f"[+] ArUco PnP viewer ({args.side} eye) on http://{host}.local:{args.port}/")
    print(f"[+] Calibration <- {calib_path}")
    print(f"[+] dict={args.dict_name} target_id={args.marker_id} marker_size={args.marker_size*1000:.0f}mm")
    print(f"[+] min_perimeter_rate={args.min_perimeter_rate} error_correction_rate={args.error_correction_rate} "
          f"upscale_steps={upscale_steps}")
    print(f"[+] confirm_hits={args.confirm_hits} max_distance_m={args.max_distance_m}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error",
                access_log=False, timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
