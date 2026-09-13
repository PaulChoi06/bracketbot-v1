import contextlib
import ctypes
import time
from pathlib import Path

import numpy as np
from bbos import Config, Reader, Type, Writer

CFG = Config("slam")

REPORT = np.dtype([
    ("timestamp_ns", np.int64),
    ("world_from_rig", np.float32, 16),
    ("vo_world_from_rig", np.float32, 16),
    ("odometry_world_from_rig", np.float32, 16),
    ("vo_head_pose", np.float32, 16),
    ("head", np.int64),
    ("tracked", np.int32),
    ("filtered", np.int32),
    ("new_landmarks", np.int32),
    ("solved", np.int32),
    ("keyframe", np.int32),
    ("pose_valid", np.int32),
    ("landmarks", np.int32),
    ("keyframes", np.int32),
    ("frame_ms", np.float32),
    ("lost", np.int32),
    ("degraded", np.int32),
    ("graph_keyframes", np.int32),
    ("history_headroom", np.int64),
    ("pgo_count", np.int64),
    ("reloc_attempts", np.int64),
    ("position", np.float32, 3),
    ("quaternion", np.float32, 4),
    ("vo_position", np.float32, 3),
    ("vo_quaternion", np.float32, 4),
], align=True)


class EngineConfig(ctypes.Structure):
    _fields_ = [("calib_yaml", ctypes.c_char_p),
                ("mask_left", ctypes.c_char_p),
                ("mask_right", ctypes.c_char_p),
                ("engine", ctypes.c_char_p),
                ("trace_path", ctypes.c_char_p),
                ("base_from_camera", ctypes.c_float * 12),
                ("planar", ctypes.c_int32),
                ("max_tracks", ctypes.c_int32),
                ("map_path", ctypes.c_char_p),
                ("map_save_interval_s", ctypes.c_float),
                ("poses_path", ctypes.c_char_p),
                ("pull_trace", ctypes.c_int32)]


class Slam:

    def __init__(self):
        lib = ctypes.CDLL(CFG.library)
        lib.bbslam_create.argtypes = [ctypes.POINTER(EngineConfig)]
        lib.bbslam_create.restype = ctypes.c_void_p
        lib.bbslam_frame.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                     ctypes.c_void_p]
        lib.bbslam_frame.restype = ctypes.c_int32
        lib.bbslam_poses_written.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                                             ctypes.POINTER(ctypes.c_int64),
                                             ctypes.POINTER(ctypes.c_int64)]
        lib.bbslam_poses_written.restype = None
        lib.bbslam_trace_pull.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int64]
        lib.bbslam_trace_pull.restype = ctypes.c_int64
        lib.bbslam_load_map.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.bbslam_load_map.restype = ctypes.c_int32
        lib.bbslam_open_history.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
        lib.bbslam_open_history.restype = ctypes.c_int64
        lib.bbslam_destroy.argtypes = [ctypes.c_void_p]
        lib.bbslam_last_error.restype = ctypes.c_char_p
        base_from_camera = np.ascontiguousarray(CFG.base_from_camera, dtype=np.float32).reshape(12)
        config = EngineConfig(
            CFG.calib.encode(), CFG.mask_left.encode(), CFG.mask_right.encode(),
            CFG.engine.encode(), None, (ctypes.c_float * 12)(*base_from_camera),
            1, 0, CFG.map_path.encode(), CFG.map_save_interval_s, CFG.poses_path.encode(),
            1 if CFG.trace else 0)
        self._lib = lib
        self._handle = lib.bbslam_create(ctypes.byref(config))
        if not self._handle:
            raise RuntimeError(lib.bbslam_last_error().decode())
        self._report = np.zeros(1, REPORT)

    def frame(self, timestamp_ns, rgb):
        status = self._lib.bbslam_frame(
            self._handle, timestamp_ns, rgb.ctypes.data_as(ctypes.c_void_p),
            self._report.ctypes.data_as(ctypes.c_void_p))
        if status != 0:
            raise RuntimeError(self._lib.bbslam_last_error().decode())
        return self._report[0]

    def poses_written(self):
        generation, count, pgo_count = ctypes.c_int64(), ctypes.c_int64(), ctypes.c_int64()
        self._lib.bbslam_poses_written(self._handle, ctypes.byref(generation),
                                       ctypes.byref(count), ctypes.byref(pgo_count))
        return generation.value, count.value, pgo_count.value

    def trace_pull(self, record, lens, payload):
        if record.nbytes != CFG.record_bytes:
            raise RuntimeError(f"slam.trace record slot is {record.nbytes} B, the engine "
                               f"writes {CFG.record_bytes} B")
        total = self._lib.bbslam_trace_pull(
            self._handle, record.ctypes.data_as(ctypes.c_void_p),
            lens.ctypes.data_as(ctypes.c_void_p),
            payload.ctypes.data_as(ctypes.c_void_p), len(payload))
        if total < 0:
            raise RuntimeError("trace pull refused")
        return total

    def load_map(self, path):
        if self._lib.bbslam_load_map(self._handle, str(path).encode()) != 0:
            raise RuntimeError(self._lib.bbslam_last_error().decode())

    def open_history(self, path, keep_existing):
        loaded = self._lib.bbslam_open_history(self._handle, str(path).encode(),
                                               1 if keep_existing else 0)
        if loaded < 0:
            raise RuntimeError(self._lib.bbslam_last_error().decode())
        return loaded

    def close(self):
        self._lib.bbslam_destroy(self._handle)


def main():
    Path(CFG.map_path).parent.mkdir(parents=True, exist_ok=True)
    map_file = Path(CFG.map_path)
    history_file = Path(CFG.history_path)
    poses_file = Path(CFG.poses_path)
    continuing = map_file.exists() and map_file.stat().st_size > 0
    if not continuing:
        # A trajectory belongs to the map it was anchored to.
        poses_file.unlink(missing_ok=True)
    engine = Slam()
    if continuing:
        engine.load_map(map_file)
        print(f"[slam] loaded {map_file}: relocalizing on boot", flush=True)
    engine.open_history(history_file, keep_existing=continuing)
    resolved = not continuing
    print(f"[slam] engine up on {CFG.input_topic}", flush=True)
    published_generation = 0
    frames = 0
    last_fed = None
    last_log = time.monotonic()
    position = np.zeros(3, np.float32)
    try:
        with Reader(CFG.input_topic, keeptime=False, sync=True) as camera, \
                Writer("slam.pose", Type("slam_pose"), keeptime=False, buf_ms=1000) as w_pose, \
                Writer("slam.history_generation", Type("slam_history_generation"),
                       keeptime=False) as w_generation, \
                Writer("slam.health", Type("slam_health"), keeptime=False) as w_health, \
                (Writer("slam.trace", Type("slam_trace"), keeptime=False, buf_ms=500)
                 if CFG.trace else contextlib.nullcontext()) as w_trace:
            while True:
                if not camera.ready():
                    time.sleep(0.001)
                    continue
                ts = int(camera.data["timestamp"])
                now = time.monotonic()
                gap_ms = 0.0 if last_fed is None else (now - last_fed) * 1e3
                last_fed = now
                report = engine.frame(ts, camera.data["rgb"])
                frames += 1
                if w_trace is not None:
                    with w_trace.buf() as b:
                        engine.trace_pull(b["record"], b["lens"], b["payload"])

                if not resolved:
                    if not bool(report["lost"]):
                        resolved = True
                        print("[slam] relocalized into the loaded map", flush=True)
                    elif int(report["reloc_attempts"]) >= CFG.boot_reloc_max_attempts:
                        print(f"[slam] boot reloc gave up after {CFG.boot_reloc_max_attempts} "
                              "attempts: fresh map", flush=True)
                        engine.close()
                        stamp = time.strftime("%Y%m%d-%H%M%S")
                        map_file.replace(map_file.with_name(f"{map_file.name}.failed-{stamp}"))
                        if history_file.exists():
                            history_file.replace(
                                history_file.with_name(f"{history_file.name}.failed-{stamp}"))
                        poses_file.unlink(missing_ok=True)
                        engine = Slam()
                        engine.open_history(history_file, keep_existing=False)
                        continuing = False
                        resolved = True
                        published_generation = 0

                if report["pose_valid"]:
                    position = np.array(report["position"])
                    with w_pose.buf() as b:
                        b["pos"] = report["position"]
                        b["quat"] = report["quaternion"]
                        b["vo_pos"] = report["vo_position"]
                        b["vo_quat"] = report["vo_quaternion"]
                        b["pgo_count"] = report["pgo_count"]
                        b["timestamp"] = ts

                generation, count, pgo_count = engine.poses_written()
                if generation != published_generation:
                    with w_generation.buf() as b:
                        b["generation"] = generation
                        b["num_poses"] = count
                        b["pgo_count"] = pgo_count
                    published_generation = generation

                with w_health.buf() as b:
                    b["degraded"] = bool(report["degraded"])
                    b["stalled"] = gap_ms > 300.0
                    b["vo_lost"] = bool(report["lost"])
                    b["last_gap_ms"] = gap_ms
                    b["localized"] = resolved
                    b["relocalized"] = continuing and resolved

                if frames % CFG.log_interval == 0:
                    now = time.monotonic()
                    print(f"[slam] f={frames} {CFG.log_interval / (now - last_log):.1f} fps "
                          f"frame={report['frame_ms']:.0f} ms kf={int(report['keyframes'])} "
                          f"lm={int(report['landmarks'])} tracked={int(report['tracked'])} "
                          f"pos=[{position[0]:.2f},{position[1]:.2f},{position[2]:.2f}] "
                          f"pgo={int(report['pgo_count'])} lost={int(report['lost'])} "
                          f"headroom={int(report['history_headroom'])}", flush=True)
                    last_log = now
    finally:
        engine.close()


if __name__ == "__main__":
    main()
