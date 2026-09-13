from pathlib import Path

import numpy as np

from bbos.registry import *          # register + the type decorators (realtime/state)
from bbos.tf import rot, trans

_runtime = Path(__file__).parent / "runtime"
_cache = Path(__file__).parent / "cache" # will fix this shit next week


@register
class depth:
    runtime_root = _runtime
    library_path = _runtime / "liblas2_depth_runtime.so"
    engine_dir = _runtime / "engines" / "las2_512_l_narrow"
    feature_engine_path = engine_dir / "feature.engine"
    cost_engine_path = engine_dir / "cost.engine"
    tail_engine_path = engine_dir / "tail.engine"
    confidence_engine_path = engine_dir / "confidence.engine"
    calib_path = _cache / "stereo_calibration_fisheye.yaml"

    input_width = 2560          
    input_height = 960
    net_width = 512             
    net_height = 384            
    out_width = net_width       
    out_height = net_height

    confidence_threshold = 0.34  
    confidence_max_disparity = 192.0
    use_density_filter = 1
    density_scale = 2.0
    mask_dilate_px = 12
    # vertical cylinder
    max_radius_m = 2.0

    min_disparity = 0.25
    min_depth_m = 0.03
    max_depth_m = 7.0
    warp_interval = 1
    point_stride = 1
    use_cuda_graph = True

    # --- extrinsic: edit these, restart depth_b, done -------------------------
    height_m = 1.55      # camera height above base
    pitch_deg = 33.0     # downward tilt
    roll_deg = -1.0      # about the camera OPTICAL AXIS -> rightmost term
    T_base_cam = (trans([0, 0, height_m]) @ rot([-1, 0, 0], 90)
                  @ rot([-1, 0, 0], pitch_deg) @ rot([0, 0, 1], roll_deg))
    camera_to_base_3x4 = T_base_cam.mat()[:3].astype(np.float32)

    # --- stock `depth` contract kept alive: tools and mapping read these ---
    width_D: int = out_width            
    height_D: int = out_height          
    downsample: float = out_width / (input_width / 2)   

    @staticmethod
    def camera_cal():
        import cv2
        """Load fisheye rectification matrices and return the scaled camera model."""
        fs = cv2.FileStorage(str(depth.calib_path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(depth.calib_path)

        mtx_l = fs.getNode("mtx_l").mat();  dist_l = fs.getNode("dist_l").mat()
        mtx_r = fs.getNode("mtx_r").mat();  dist_r = fs.getNode("dist_r").mat()
        R1 = fs.getNode("R1").mat();        R2 = fs.getNode("R2").mat()
        P1 = fs.getNode("P1").mat();        P2 = fs.getNode("P2").mat()
        R = fs.getNode("R").mat();         t = fs.getNode("T").mat().astype(np.float32).squeeze() / 1000.0
        Q  = fs.getNode("Q").mat().astype(np.float32)
        fs.release()
        # Scale translation component to match *scale*
        Q[:4, 3] *= depth.downsample
        baseline_m = abs(P2[0, 3] / P2[0, 0]) / 1000.0
        fx_ds = P1[0, 0] * depth.downsample
        return mtx_l, dist_l, mtx_r, dist_r, R1, R2, P1, P2, Q, baseline_m, fx_ds, R, t


@register
class points:
    num_points = (-(-depth.width_D // depth.point_stride)) * (-(-depth.height_D // depth.point_stride))


@realtime(ms=100)
def camera_depth():
    return [
        ("depth", np.uint16, (depth.height_D, depth.width_D)),
        ("depth_raw", np.uint16, (depth.height_D, depth.width_D)),
    ]


@realtime(ms=100)
def camera_points():
    return [
        ("num_points", np.int32),
        ("points", np.float16, (points.num_points, 3)),
        ("colors", np.uint8, (points.num_points, 3)),
        ("idx_2d", np.int32, (points.num_points,)),
    ]

@realtime(ms=100)
def camera_rect():
    return [
        ("left", np.uint8, (depth.height_D, depth.width_D, 3)),
    ]
