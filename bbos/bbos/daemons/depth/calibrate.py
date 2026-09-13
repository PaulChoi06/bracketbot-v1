#!/usr/bin/env python3
"""
Depth stereo calibration installer.

Downloads a calibration JSON from Google Drive, converts it to OpenCV YAML
(fisheye model), and installs both files into the depth daemon cache.

Run via:  calibrate depth
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

CACHE_DIR = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# Ensure gdown is available
# ---------------------------------------------------------------------------

def ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("[i] Installing gdown ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown>=4.7.0"])
        import gdown
        return gdown


# ---------------------------------------------------------------------------
# JSON -> OpenCV YAML converter (fisheye stereo calibration)
# ---------------------------------------------------------------------------

def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for key in path.split("/"):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _cam_params(cam):
    prm = _get(cam, "model/ptr_wrapper/data/parameters")
    if prm is None:
        raise KeyError("Missing camera parameters in JSON at model/ptr_wrapper/data/parameters")
    f  = float(prm["f"]["val"])
    ar = float(prm.get("ar", {}).get("val", 1.0) or 1.0)
    cx = float(prm["cx"]["val"])
    cy = float(prm["cy"]["val"])
    k1 = float(prm["k1"]["val"])
    k2 = float(prm["k2"]["val"])
    k3 = float(prm.get("k3", {}).get("val", 0.0) or 0.0)
    k4 = float(prm.get("k4", {}).get("val", 0.0) or 0.0)
    return f, ar, cx, cy, (k1, k2, k3, k4)


def _rodrigues_to_R(rvec):
    import cv2
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    try:
        R, _ = cv2.Rodrigues(rvec)
        return np.asarray(R, dtype=np.float64)
    except Exception:
        pass
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = (rvec / theta).reshape(3)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _write_cv_yaml(path: str, mats: Dict[str, np.ndarray]):
    def mat_tag(name, arr):
        arr = np.asarray(arr, dtype=np.float64)
        rows, cols = arr.shape
        flat = ", ".join(f"{x:.17g}" for x in arr.reshape(-1))
        return (
            f"{name}: !!opencv-matrix\n"
            f"   rows: {rows}\n"
            f"   cols: {cols}\n"
            f"   dt: d\n"
            f"   data: [ {flat} ]\n"
        )

    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        for k in ["mtx_l", "dist_l", "mtx_r", "dist_r", "R", "T", "R1", "R2", "P1", "P2", "Q"]:
            f.write(mat_tag(k, mats[k]))


def convert(in_json: str, out_yaml: str, balance=0.0, fov_scale=1.0, zero_disparity=True):
    import cv2

    with open(in_json, "r") as f:
        data = json.load(f)
    cal = data.get("Calibration") or data.get("calibration") or data
    cams = cal["cameras"]
    if not isinstance(cams, list) or len(cams) < 2:
        raise ValueError("Expected at least two cameras in Calibration.cameras")

    f_l, ar_l, cx_l, cy_l, kd_l = _cam_params(cams[0])
    f_r, ar_r, cx_r, cy_r, kd_r = _cam_params(cams[1])

    img_size = (
        _get(cams[0], "model/ptr_wrapper/data/CameraModelCRT/CameraModelBase/imageSize")
        or _get(cams[0], "model/ptr_wrapper/data/CameraModelBase/imageSize")
    )
    if img_size is None:
        raise KeyError("Missing image size in JSON (CameraModelBase/imageSize)")
    w, h = int(img_size["width"]), int(img_size["height"])

    fx_l, fy_l = f_l, f_l * ar_l
    fx_r, fy_r = f_r, f_r * ar_r
    K1 = np.array([[fx_l, 0., cx_l], [0., fy_l, cy_l], [0., 0., 1.]], dtype=np.float64)
    K2 = np.array([[fx_r, 0., cx_r], [0., fy_r, cy_r], [0., 0., 1.]], dtype=np.float64)
    D1 = np.array(kd_l, dtype=np.float64).reshape(4, 1)
    D2 = np.array(kd_r, dtype=np.float64).reshape(4, 1)

    rx = float(_get(cams[1], "transform/rotation/rx", 0.0))
    ry = float(_get(cams[1], "transform/rotation/ry", 0.0))
    rz = float(_get(cams[1], "transform/rotation/rz", 0.0))
    R = _rodrigues_to_R(np.array([rx, ry, rz]))

    tx = float(_get(cams[1], "transform/translation/x", 0.0))
    ty = float(_get(cams[1], "transform/translation/y", 0.0))
    tz = float(_get(cams[1], "transform/translation/z", 0.0))
    T_mm = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1) * 1000.0

    flags = cv2.CALIB_ZERO_DISPARITY if zero_disparity else 0
    if hasattr(cv2, "fisheye") and hasattr(cv2.fisheye, "stereoRectify"):
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K1, D1, K2, D2, (w, h), R, T_mm,
            flags=flags, balance=float(balance), fov_scale=float(fov_scale),
        )
    else:
        Z = np.zeros((1, 5), dtype=np.float64)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, Z, K2, Z, (w, h), R, T_mm, flags=flags, alpha=0.0,
        )

    _write_cv_yaml(out_yaml, {
        "mtx_l": K1, "dist_l": D1, "mtx_r": K2, "dist_r": D2,
        "R": R, "T": T_mm, "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DEPTH STEREO CALIBRATION")
    print("=" * 60)
    print()
    print("This will download a stereo calibration JSON from Google Drive,")
    print("convert it to OpenCV YAML (fisheye), and install it into:")
    print(f"  {CACHE_DIR}/")
    print()

    url = input("Paste Google Drive link to calibration JSON: ").strip()
    if not url:
        print("No URL provided. Aborting.")
        sys.exit(1)

    gdown = ensure_gdown()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        json_path = tmp / "calibration.json"
        yaml_path = tmp / "stereo_calibration_fisheye.yaml"

        # Download
        print(f"\n[i] Downloading from: {url}")
        gdown.download(url, str(json_path), quiet=False)
        if not json_path.exists() or json_path.stat().st_size == 0:
            print("[!] Download failed or produced empty file.", file=sys.stderr)
            sys.exit(1)
        print(f"[+] Downloaded -> {json_path}")

        # Convert
        print("[i] Converting JSON -> YAML ...")
        convert(str(json_path), str(yaml_path))
        if not yaml_path.exists() or yaml_path.stat().st_size == 0:
            print("[!] Conversion produced no output.", file=sys.stderr)
            sys.exit(1)
        print(f"[+] Converted -> {yaml_path}")

        # Install
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        final_yaml = CACHE_DIR / "stereo_calibration_fisheye.yaml"
        final_json = CACHE_DIR / "stereo_calibration_fisheye.json"
        shutil.copy2(yaml_path, final_yaml)
        shutil.copy2(json_path, final_json)
        print(f"[+] Installed YAML -> {final_yaml}")
        print(f"[+] Stored JSON  -> {final_json}")

    print()
    print("=" * 60)
    print("DEPTH CALIBRATION COMPLETE")
    print("=" * 60)
    print("Restart the depth daemon to pick up the new calibration.")


if __name__ == "__main__":
    main()