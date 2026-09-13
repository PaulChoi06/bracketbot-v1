"""depth daemon: publishes camera.points + camera.depth + camera.rect"""

import contextlib
import ctypes
import os
import time

# MPS currently not in use 
#if os.path.exists("/tmp/nvidia-mps/control"):
#    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
#    os.environ.setdefault("CUDA_MPS_CLIENT_PRIORITY", "1")  # slam runs at 0

import numpy as np

from bbos import Reader, Writer, Type, Config

C = Config("depth")
_LIB = ctypes.CDLL(str(C.library_path))


class Cfg(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("calibration_path", ctypes.c_char_p),
        ("feature_engine_path", ctypes.c_char_p), ("cost_engine_path", ctypes.c_char_p),
        ("tail_engine_path", ctypes.c_char_p), ("confidence_engine_path", ctypes.c_char_p),
        ("input_width", ctypes.c_int32), ("input_height", ctypes.c_int32),
        ("output_width", ctypes.c_int32), ("output_height", ctypes.c_int32),
        ("warp_interval", ctypes.c_int32), ("point_stride", ctypes.c_int32),
        ("use_cuda_graph", ctypes.c_int32),
        ("min_disparity", ctypes.c_float), ("min_depth_m", ctypes.c_float),
        ("max_depth_m", ctypes.c_float), ("confidence_max_disparity", ctypes.c_float),
        ("confidence_threshold", ctypes.c_float),
        ("camera_to_base_3x4", ctypes.POINTER(ctypes.c_float)),
        ("use_density_filter", ctypes.c_int32),
        ("density_scale", ctypes.c_float),
        ("mask_dilate_px", ctypes.c_int32),
        ("max_radius_m", ctypes.c_float),
    ]


_LIB.las2_depth_create.restype = ctypes.c_void_p
_LIB.las2_depth_create.argtypes = [ctypes.POINTER(Cfg), ctypes.c_char_p, ctypes.c_size_t]
_LIB.las2_depth_set_camera_to_base.restype = ctypes.c_int32
_LIB.las2_depth_set_camera_to_base.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                               ctypes.c_char_p, ctypes.c_size_t]
_LIB.las2_depth_process_rgb.restype = ctypes.c_int32
_LIB.las2_depth_process_rgb.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                                        ctypes.c_int32, ctypes.c_int32, ctypes.c_uint64,
                                        ctypes.c_char_p, ctypes.c_size_t]
for fn, rt in [("point_count", ctypes.c_uint32),
               ("max_points", ctypes.c_uint32),
               ("points_base", ctypes.POINTER(ctypes.c_float)),
               ("point_colors_rgb", ctypes.POINTER(ctypes.c_uint8)),
               ("point_idx_2d", ctypes.POINTER(ctypes.c_int32)),
               ("depth_m", ctypes.POINTER(ctypes.c_float)),
               ("depth_raw_m", ctypes.POINTER(ctypes.c_float)),
               ("left_rgb", ctypes.POINTER(ctypes.c_uint8))]:
    f = getattr(_LIB, "las2_depth_" + fn)
    f.restype = rt
    f.argtypes = [ctypes.c_void_p]
_LIB.las2_depth_output_timestamp_ns.restype = ctypes.c_uint64
_LIB.las2_depth_output_timestamp_ns.argtypes = [ctypes.c_void_p]

_views = {}


def view(ptr, shape):
    """ndarray view over a C buffer, built once per (address, shape)."""
    key = (ctypes.addressof(ptr.contents), shape)
    v = _views.get(key)
    if v is None:
        v = np.ctypeslib.as_array(ptr, shape)
        _views[key] = v
    return v


PERIOD_NS = int(Type("camera_points")()[1]) * 1_000_000

MW, MH = C.net_width, C.net_height      # lib canvas == engine input dims
OW = OH = None


def main():
    t_bc = np.ascontiguousarray(C.camera_to_base_3x4, np.float32).reshape(12)
    cfg = Cfg(
        abi_version=7,
        calibration_path=str(C.calib_path).encode(),
        feature_engine_path=str(C.feature_engine_path).encode(),
        cost_engine_path=str(C.cost_engine_path).encode(),
        tail_engine_path=str(C.tail_engine_path).encode(),
        confidence_engine_path=str(C.confidence_engine_path).encode(),
        input_width=C.input_width, input_height=C.input_height,
        output_width=MW, output_height=MH,
        warp_interval=C.warp_interval, point_stride=C.point_stride,
        use_cuda_graph=int(C.use_cuda_graph),
        min_disparity=C.min_disparity, min_depth_m=C.min_depth_m,
        max_depth_m=C.max_depth_m,
        confidence_max_disparity=C.confidence_max_disparity,
        confidence_threshold=C.confidence_threshold,
        camera_to_base_3x4=t_bc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        use_density_filter=C.use_density_filter,
        density_scale=C.density_scale,
        mask_dilate_px=C.mask_dilate_px,
        max_radius_m=C.max_radius_m,
    )
    err = ctypes.create_string_buffer(2048)
    with contextlib.ExitStack() as stack:
        w_points = stack.enter_context(Writer("camera.points", Type("camera_points"), buf_ms=200))
        w_depth = stack.enter_context(Writer("camera.depth", Type("camera_depth")))
        w_rect = stack.enter_context(Writer("camera.rect", Type("camera_rect"), buf_ms=500))
        global OW, OH
        with w_points.buf() as b:
            MAXPTS = int(b["points"].shape[0])
        with w_rect.buf() as b:
            OH, OW = int(b["left"].shape[0]), int(b["left"].shape[1])
        if (OW, OH) != (MW, MH):
            raise RuntimeError(f"depth_b: camera.rect is {OW}x{OH} but the runtime canvas is "
                               f"{MW}x{MH}; out_width/out_height must equal net_width/net_height")
        with w_depth.buf() as b:
            DH, DW = int(b["depth"].shape[0]), int(b["depth"].shape[1])
        if (DW, DH) != (MW, MH):
            raise RuntimeError(f"depth_b: camera.depth is {DW}x{DH}, expected {MW}x{MH}")

        handle = _LIB.las2_depth_create(ctypes.byref(cfg), err, 2048)
        if not handle:
            raise RuntimeError(f"depth_b: create failed: {err.value.decode(errors='replace')}")
        print(f"[depth_b] {os.path.basename(str(C.engine_dir))} {MW}x{MH} "
              f"conf={C.confidence_threshold} density={C.use_density_filter} "
              f"dscale={C.density_scale} maskdil={C.mask_dilate_px} rmax={C.max_radius_m} "
              f"h={C.height_m} pitch={C.pitch_deg} roll={C.roll_deg} imupitch={getattr(C, 'imu_pitch', 0)} "
              f"maxpts={MAXPTS} -> camera.points + camera.depth/rect", flush=True)

        CAP = int(_LIB.las2_depth_max_points(handle))
        if CAP != MAXPTS:
            raise RuntimeError(f"depth_b: runtime emits up to {CAP} points but camera.points holds "
                               f"{MAXPTS}; points.num_points must match max_points_ in the .so "
                               f"(both are ceil(w/point_stride)*ceil(h/point_stride))")
        depth_u16 = depth_raw_u16 = rect_o = s_pts = s_cols = s_idx = None
        out_ts = 0
        r_cam = Reader("camera.head.rgb", sync=False)
        r_imu = Reader("imu.orientation", sync=False) if getattr(C, "imu_pitch", 0) else None
        try:
            DRIVE_SIGN = float(Config("drive").sign_pitch)
        except Exception:
            DRIVE_SIGN = -1.0
        t_bc_base = np.array(C.camera_to_base_3x4, np.float32).reshape(3, 4)
        imu_ref = None
        imu_applied = 0.0
        t_live = np.ascontiguousarray(t_bc_base, np.float32).reshape(12)
        next_ts = 0
        fresh = False
        while True:
            fresh = False
            if r_imu is not None and r_imu.ready() and r_imu.data is not None:
                cur = DRIVE_SIGN * float(np.asarray(r_imu.data["rpy"])[int(C.imu_pitch_idx)])
                if imu_ref is None:
                    imu_ref = cur
                    print(f"[depth_b] imu pitch ref={imu_ref:+.3f} deg "
                          f"(idx {int(C.imu_pitch_idx)}, sign {float(C.imu_pitch_sign):+.0f})", flush=True)
                d = float(C.imu_pitch_sign) * (cur - imu_ref)
                lim = float(C.imu_pitch_max_deg)
                d = max(-lim, min(lim, d))
                if abs(d - imu_applied) > 0.05:
                    th = np.radians(d); c_, s_ = np.cos(th), np.sin(th)
                    M = t_bc_base.copy()
                    M[:, :3] = M[:, :3] @ np.array([[1.0, 0.0, 0.0],
                                                    [0.0,  c_,  s_],
                                                    [0.0, -s_,  c_]], np.float32)
                    t_live = np.ascontiguousarray(M, np.float32).reshape(12)
                    _LIB.las2_depth_set_camera_to_base(
                        handle, t_live.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), err, 2048)
                    imu_applied = d
            if r_cam.ready():
                ts = int(r_cam.data["timestamp"].view("i8"))
                rgb = r_cam.data["rgb"]
                if rgb.size and ts >= next_ts:
                    next_ts = ts + PERIOD_NS
                    buf = np.ascontiguousarray(rgb)
                    bp = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                    if _LIB.las2_depth_process_rgb(handle, bp, buf.shape[1], buf.shape[0],
                                                   ts, err, 2048) > 0:
                        npc = int(_LIB.las2_depth_point_count(handle))
                        pts = view(_LIB.las2_depth_points_base(handle), (CAP, 3))[:npc]
                        cols = view(_LIB.las2_depth_point_colors_rgb(handle), (CAP, 3))[:npc]
                        midx = view(_LIB.las2_depth_point_idx_2d(handle), (CAP,))[:npc]
                        depth = view(_LIB.las2_depth_depth_m(handle), (MH, MW))
                        depth_raw = view(_LIB.las2_depth_depth_raw_m(handle), (MH, MW))
                        left = view(_LIB.las2_depth_left_rgb(handle), (MH, MW, 3))
                        out_ts = int(_LIB.las2_depth_output_timestamp_ns(handle))
                        depth_u16 = (np.minimum(depth, 60.0) * 1000).astype(np.uint16)
                        depth_raw_u16 = (np.minimum(depth_raw, 60.0) * 1000).astype(np.uint16)
                        rect_o = left
                        s_pts = pts.astype(np.float16)
                        s_cols = cols
                        s_idx = midx
                        fresh = True
            if fresh:
                with w_rect.buf() as b:
                    b["left"] = rect_o
                    b["timestamp"] = np.datetime64(out_ts, "ns")
                with w_points.buf() as b:
                    k = len(s_pts)
                    b["num_points"] = k
                    b["points"][:k] = s_pts
                    b["colors"][:k] = s_cols
                    b["idx_2d"][:k] = s_idx
                    b["timestamp"] = np.datetime64(out_ts, "ns")
                with w_depth.buf() as b:
                    b["depth"] = depth_u16
                    b["depth_raw"] = depth_raw_u16
                    b["timestamp"] = np.datetime64(out_ts, "ns")
            else:
                time.sleep(0.002)


if __name__ == "__main__":
    main()
