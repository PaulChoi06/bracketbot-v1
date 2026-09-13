"""The mapping daemon: pairs every depth frame with it's visual odometery pose, and computes a 3d map of the environment

On a Pose Graph Optimization (PGO), the mapping daemon rebuilds the map from, and publishes the voxel cloud, and the 2d nav grid."""
import ctypes
import math
import os
import shutil
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
from bbos import Config, Reader, Type, Writer

CFG = Config("mapping")
DEPTH_CFG = Config("depth")
SLAM_CFG = Config("slam")
GRID2D_PERIOD_S = Type("mapping_grid2d")()[1] * 1e-3  # the publish cadences are the topics' declared periods
VOXELS_PERIOD_S = Type("mapping_voxels")()[1] * 1e-3

FLOAT, INT = ctypes.c_float, ctypes.c_int
FLOAT_ARRAY = np.ctypeslib.ndpointer(np.float32, flags="C")
U16_ARRAY = np.ctypeslib.ndpointer(np.uint16, flags="C")
U8_ARRAY = np.ctypeslib.ndpointer(np.uint8, flags="C")
I32_ARRAY = np.ctypeslib.ndpointer(np.int32, flags="C")
I64_ARRAY = np.ctypeslib.ndpointer(np.int64, flags="C")
U64_ARRAY = np.ctypeslib.ndpointer(np.uint64, flags="C")


def load_library(path):
    lib = ctypes.CDLL(str(path))
    lib.map_create.restype = ctypes.c_void_p
    lib.map_create.argtypes = [INT, INT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT_ARRAY]
    lib.map_calib.argtypes = [ctypes.c_char_p, INT, INT, FLOAT_ARRAY, ctypes.c_void_p]
    lib.map_segmenter_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.map_segment.argtypes = [ctypes.c_void_p, ctypes.c_void_p, INT, ctypes.c_int64]
    lib.map_integrate_seg.argtypes = [ctypes.c_void_p, ctypes.c_uint32, FLOAT_ARRAY, U16_ARRAY, U16_ARRAY, U8_ARRAY, ctypes.c_int64, FLOAT_ARRAY]
    lib.map_set_keyframes.argtypes = [ctypes.c_void_p, I64_ARRAY, FLOAT_ARRAY, FLOAT_ARRAY, INT]
    lib.map_rebuild_pgo_start.argtypes = [ctypes.c_void_p]
    lib.map_rebuild_poll.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.map_set_tile_dir.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.map_clear.argtypes = [ctypes.c_void_p]
    lib.map_flush_dirty.argtypes = [ctypes.c_void_p, INT]
    lib.map_tile_counts.argtypes = [ctypes.c_void_p, I32_ARRAY, I32_ARRAY, I32_ARRAY]
    lib.map_save_frames.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.map_load_frames.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.map_page.argtypes = [ctypes.c_void_p, FLOAT, FLOAT, FLOAT, FLOAT, ctypes.c_uint32, INT, ctypes.c_uint32, INT]
    lib.map_tile_list.restype = INT
    lib.map_tile_list.argtypes = [ctypes.c_void_p, I32_ARRAY, INT]
    lib.map_tile_export.restype = INT
    lib.map_tile_export.argtypes = [ctypes.c_void_p, INT, INT, FLOAT_ARRAY, U8_ARRAY, U8_ARRAY, INT]
    lib.map_tile_export_info.restype = INT
    lib.map_tile_export_info.argtypes = [ctypes.c_void_p, INT, INT, I32_ARRAY, INT]
    lib.map_tile_list_disk.restype = INT
    lib.map_tile_list_disk.argtypes = [ctypes.c_void_p, I32_ARRAY, INT]
    lib.map_rebuild_diff.restype = INT
    lib.map_rebuild_diff.argtypes = [ctypes.c_void_p, FLOAT_ARRAY, INT, FLOAT_ARRAY, INT, FLOAT_ARRAY, INT, I32_ARRAY]
    lib.map_tile_capacity.argtypes = [I32_ARRAY, I32_ARRAY]
    lib.map_grid.restype = INT
    lib.map_grid.argtypes = [ctypes.c_void_p, INT, FLOAT, FLOAT, FLOAT, U8_ARRAY, FLOAT_ARRAY]
    lib.map_tile_export_holes.restype = INT
    lib.map_tile_export_holes.argtypes = [ctypes.c_void_p, INT, INT, FLOAT_ARRAY, I32_ARRAY, INT]
    lib.map_tile_dirty.restype = INT
    lib.map_tile_dirty.argtypes = [ctypes.c_void_p, INT, INT]
    lib.map_tile_clean.argtypes = [ctypes.c_void_p, INT, INT]
    lib.map_counters.argtypes = [ctypes.c_void_p, U64_ARRAY]
    return lib


# slam's corrected trajectory file: a 40-byte header (magic, record size, record count,
# generation, pgo count), then one record per keyframe, oldest first, rewritten atomically
POSES_DTYPE = np.dtype([("ts_ns", "<i8"), ("pos", "<f4", 3), ("quat", "<f4", 4), ("pad", "u1", 4)])


class SlamHistory(NamedTuple):
    timestamps_ns: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    generation: int
    pgo_count: int


def read_slam_history(path):
    try:
        with open(path, "rb") as file:
            raw = file.read()
    except OSError:
        return None
    if len(raw) < 40 or raw[:8] != b"BBPOSE1\0":
        return None
    record_size, record_count, generation, pgo_count = np.frombuffer(raw[8:40], "<i8").tolist()
    if record_size != POSES_DTYPE.itemsize or len(raw) < 40 + record_count * record_size:
        return None
    rows = np.frombuffer(raw[40:40 + record_count * record_size], POSES_DTYPE)
    return SlamHistory(rows["ts_ns"], rows["pos"], rows["quat"], generation, pgo_count)


def yaw_of(quat_xyzw):
    return 2.0 * math.atan2(float(quat_xyzw[2]), float(quat_xyzw[3]))


# a planar pose as the 3x4 world_from_base matrix the library takes
def pose_matrix(x, y, yaw):
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return np.array([cos_yaw, -sin_yaw, 0, x, sin_yaw, cos_yaw, 0, y, 0, 0, 1, 0], np.float32)


def timestamp_ns(message):
    return int(message["timestamp"].view("i8"))


class Export:
    # The published cloud, kept per tile so that only the tiles that changed since their last
    # export are read from the map again. A tile paged to disk keeps publishing its last arrays.
    def __init__(self, lib):
        capacity = np.zeros(2, np.int32)
        lib.map_tile_capacity(capacity[0:1], capacity[1:2])
        self.max_voxels, self.max_columns = int(capacity[0]), int(capacity[1])
        self.tile_list = np.zeros((4096, 2), np.int32)
        self.disk_list = np.zeros((4096, 2), np.int32)
        # one tile's export lands here, then is copied into the per-tile dicts
        self.xyz = np.zeros((self.max_voxels, 3), np.float32)
        self.rgba = np.zeros((self.max_voxels, 4), np.uint8)
        self.flags = np.zeros(self.max_voxels, np.uint8)
        self.info = np.zeros((self.max_voxels, 4), np.int32)
        self.hole_xy = np.zeros((self.max_columns, 2), np.float32)
        self.hole_info = np.zeros((self.max_columns, 3), np.int32)
        self.tiles = {}  # (tile_x, tile_y) -> (xyz, rgb, flags, info) as last exported
        self.holes = {}  # (tile_x, tile_y) -> (hole_xy, hole_info)
        self.tile_count = 0

    def refresh(self, lib, map_handle):
        self.tile_count = lib.map_tile_list(map_handle, self.tile_list, len(self.tile_list))
        refreshed = 0
        for tile_x, tile_y in self.tile_list[:self.tile_count].tolist():
            if (tile_x, tile_y) in self.tiles and not lib.map_tile_dirty(map_handle, tile_x, tile_y):
                continue
            refreshed += 1
            voxel_count = lib.map_tile_export(map_handle, tile_x, tile_y, self.xyz, self.rgba, self.flags, self.max_voxels)
            lib.map_tile_export_info(map_handle, tile_x, tile_y, self.info, self.max_voxels)
            self.tiles[(tile_x, tile_y)] = (self.xyz[:voxel_count].copy(), self.rgba[:voxel_count, :3].copy(),
                                            self.flags[:voxel_count].copy(), self.info[:voxel_count].copy())
            hole_count = lib.map_tile_export_holes(map_handle, tile_x, tile_y, self.hole_xy, self.hole_info, self.max_columns)
            self.holes[(tile_x, tile_y)] = (self.hole_xy[:hole_count].copy(), self.hole_info[:hole_count].copy())
            lib.map_tile_clean(map_handle, tile_x, tile_y)
        # tiles the map no longer has, in memory or on disk, leave the published cloud
        disk_count = lib.map_tile_list_disk(map_handle, self.disk_list, len(self.disk_list))
        known = set(map(tuple, self.tile_list[:self.tile_count].tolist())) | set(map(tuple, self.disk_list[:disk_count].tolist()))
        for key in [key for key in self.tiles if key not in known]:
            del self.tiles[key]
            self.holes.pop(key, None)
        return refreshed

    def cloud(self):
        if not self.tiles:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros(0, np.uint8), np.zeros((0, 4), np.int32)
        parts = list(self.tiles.values())
        return (np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts]),
                np.concatenate([part[2] for part in parts]), np.concatenate([part[3] for part in parts]))



# slam publishes the pose of a frame 150-350 ms after depth does: wait for the pose whose
# timestamp is within pose_gap_ns of the depth frame, up to wait_s. None if it never comes
def wait_for_pose(pose_in, depth_ts, pose_gap_ns, wait_s):
    started = time.perf_counter()
    while True:
        pose_in.ready()
        pose = pose_in.data
        gap = abs(timestamp_ns(pose) - depth_ts) if pose is not None else None
        if gap is not None and gap <= pose_gap_ns:
            return pose
        if time.perf_counter() - started > wait_s:
            return None
        time.sleep(0.002)


# (localized, relocalized) from slam.health once slam reports localized, (False, None) if it does
# not within wait_s, (None, None) if the topic is unavailable
def slam_provenance(wait_s):
    started = time.monotonic()
    try:
        with Reader("slam.health", keeptime=False) as health_in:
            while time.monotonic() - started < wait_s:
                health_in.ready()
                health = health_in.data
                if health is not None and bool(health["localized"]):
                    return True, bool(health["relocalized"])
                time.sleep(0.2)
    except Exception as error:
        print(f"[mapping] slam.health unavailable ({error}): keeping the map", flush=True)
        return None, None
    return False, None


# The map under map_dir: continued when slam relocalized into its saved map, dropped when slam
# started a fresh one (its frame is gone, so ours is). Returns the frame index to continue with
def open_map(lib, map_handle):
    map_dir = Path(CFG.map_dir)
    localized, relocalized = slam_provenance(CFG.slam_health_wait_s)
    if localized is True and relocalized is False:
        print(f"[mapping] slam booted into a fresh map: dropping {map_dir}", flush=True)
        shutil.rmtree(map_dir, ignore_errors=True)
    elif localized is True:
        print("[mapping] slam relocalized into its saved map: continuing ours", flush=True)
    elif localized is False:
        print(f"[mapping] slam did not report localized within {CFG.slam_health_wait_s:.0f}s: keeping the map, "
              "integration stays gated on the flag", flush=True)
    frame = lib.map_load_frames(map_handle, str(map_dir / "frames.bin").encode())
    if frame < 0:
        print(f"[mapping] {map_dir / 'frames.bin'} unreadable: dropping the map", flush=True)
        shutil.rmtree(map_dir, ignore_errors=True)
        frame = 0
    (map_dir / "tiles").mkdir(parents=True, exist_ok=True)
    tiles_on_disk = lib.map_set_tile_dir(map_handle, str(map_dir / "tiles").encode())
    print(f"[mapping] map {map_dir}: {frame} frames, {tiles_on_disk} tiles on disk", flush=True)
    return frame, localized, relocalized


# slam started a fresh map while we ran: ours goes, in memory and on disk
def reset_map(lib, map_handle):
    map_dir = Path(CFG.map_dir)
    print(f"[mapping] slam started a fresh map: dropping {map_dir}", flush=True)
    lib.map_clear(map_handle)
    shutil.rmtree(map_dir, ignore_errors=True)
    (map_dir / "tiles").mkdir(parents=True, exist_ok=True)
    lib.map_set_tile_dir(map_handle, str(map_dir / "tiles").encode())


TILE_COUNTS = np.zeros(3, np.int32)


def save_map(lib, map_handle):
    started = time.perf_counter()
    tiles_written = lib.map_flush_dirty(map_handle, 1 << 20)
    status = lib.map_save_frames(map_handle, str(Path(CFG.map_dir) / "frames.bin").encode())
    lib.map_tile_counts(map_handle, TILE_COUNTS[0:1], TILE_COUNTS[1:2], TILE_COUNTS[2:3])
    return tiles_written, status, 1e3 * (time.perf_counter() - started)


def main():
    width, height = int(DEPTH_CFG.width_D), int(DEPTH_CFG.height_D)
    run_dir = Path(CFG.debug_output_dir) / time.strftime("%Y%m%d_%H%M%S")
    (run_dir / "frames").mkdir(parents=True)
    if CFG.debug_output:
        os.environ["MAP_DUMP"] = str(run_dir / "dump")
    lib = load_library(Path(CFG.library))
    calib_path = str(DEPTH_CFG.calib_path).encode()
    intrinsics = np.zeros(4, np.float32)
    assert lib.map_calib(calib_path, width, height, intrinsics, None) == 0, "calibration unreadable"
    fx, fy, cx, cy = intrinsics.tolist()
    camera_to_base = np.ascontiguousarray(np.asarray(DEPTH_CFG.camera_to_base_3x4, np.float32)).ravel()
    map_handle = lib.map_create(width, height, fx, fy, cx, cy, 0.001, camera_to_base)
    assert map_handle, "map_create failed"
    frame, slam_localized, slam_relocalized = open_map(lib, map_handle)  # frame: the index the map knows the next frame by
    # what slam last reported, to notice it starting over: localized coming back, relocalized
    # dropping, or the history generation going backwards
    was_localized = slam_localized is True
    was_fresh = slam_localized is True and slam_relocalized is False
    last_generation = None
    assert lib.map_segmenter_init(map_handle, str(CFG.engine).encode(), calib_path) == 0, "segmenter init failed"
    print(f"[mapping] run {run_dir}  intrinsics fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}", flush=True)
    stop = threading.Event()  # SIGTERM from the manager: save the map, then leave
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    robot_pose = None  # (x, y, yaw) of the last integrated frame
    previous_pose, previous_pose_at = None, None  # for the speed the pager prefetches with
    dropped_no_pose = dropped_no_rgb = dropped_slam_lost = 0
    segmented = 0
    integrate_times = deque(maxlen=200)
    pose_waits = deque(maxlen=200)
    counters = np.zeros(8, np.uint64)
    left_rgb = np.zeros((height, width, 3), np.uint8)
    pose_gap_ns = int(CFG.pose_gap_ms * 1e6)
    pose_wait_s = CFG.pose_wait_ms * 1e-3
    mask_gap_ns = int(CFG.mask_gap_ms * 1e6)
    # the rebuild in flight, if any
    keyframe_count = 0
    rebuild_started_at = None
    rebuild_pgo_count = 0
    rebuilt_generation = None
    last_rebuild_at = -1e9
    pgo_pending = False
    # publishing
    export = Export(lib)
    grid2d_size = int(CFG.grid2d_size)
    grid2d = np.zeros((grid2d_size, grid2d_size), np.uint8)
    grid2d_origin = np.zeros(2, np.float32)
    published_voxels = 0
    last_publish_at = last_cloud_at = 0.0
    last_save_at = time.monotonic()
    last_log_at = time.monotonic()
    last_lost_log_at = 0.0
    # where the loop's time goes, per stage, between log lines
    stage_seconds = {}
    stage_started = time.perf_counter()
    frames_since_log = 0

    def end_stage(name):
        nonlocal stage_started
        now = time.perf_counter()
        stage_seconds[name] = stage_seconds.get(name, 0.0) + (now - stage_started)
        stage_started = now

    with Writer("mapping.voxels", Type("mapping_voxels"), keeptime=False) as voxels_out, \
         Writer("mapping.grid2d", Type("mapping_grid2d"), keeptime=False) as grid2d_out, \
         Writer("mapping.reproject", Type("mapping_reproject"), keeptime=False) as reproject_out, \
         Writer("mapping.rebuild", Type("mapping_rebuild"), keeptime=False) as rebuild_out, \
         Reader("camera.depth", keeptime=True) as depth_in, \
         Reader("camera.head.rgb", keeptime=False, aligned_to=depth_in) as rgb_in, \
         Reader("camera.rect", keeptime=False, aligned_to=depth_in) as rect_in, \
         Reader("slam.pose", keeptime=False, aligned_to=depth_in) as pose_in, \
         Reader("slam.history_generation", keeptime=False) as generation_in, \
         Reader("slam.health", keeptime=False) as health_in:
        with reproject_out.buf() as out:
            out["reprojecting"] = False
        while not stop.is_set():
            health_in.ready()
            health = health_in.data
            if depth_in.ready():
                depth_frame = depth_in.data
                depth_ts = timestamp_ns(depth_frame)
                end_stage("idle")
                # slam lost or still relocalizing: its pose is not in the map frame, nothing goes in
                # (checked before waiting for a pose that would not come)
                if health is not None and (bool(health["vo_lost"]) or not bool(health["localized"])):
                    dropped_slam_lost += 1
                    if time.monotonic() - last_lost_log_at >= CFG.log_interval_s:
                        last_lost_log_at = time.monotonic()
                        print(f"[mapping] slam vo_lost={bool(health['vo_lost'])} localized={bool(health['localized'])}: "
                              f"nothing integrated ({dropped_slam_lost} frames dropped so far)", flush=True)
                    continue
                wait_started = time.perf_counter()
                pose = wait_for_pose(pose_in, depth_ts, pose_gap_ns, pose_wait_s)
                if pose is None:
                    dropped_no_pose += 1
                    continue
                pose_waits.append(time.perf_counter() - wait_started)
                end_stage("pose_wait")
                robot_pose = (float(pose["pos"][0]), float(pose["pos"][1]), yaw_of(pose["quat"]))
                vo_pose = (float(pose["vo_pos"][0]), float(pose["vo_pos"][1]), yaw_of(pose["vo_quat"]))
                # the camera image of the same instant goes to the segmenter
                rgb_in.ready()
                rgb_frame = rgb_in.data
                if rgb_frame is None or abs(timestamp_ns(rgb_frame) - depth_ts) > mask_gap_ns:
                    dropped_no_rgb += 1
                    continue
                rgb_ts = timestamp_ns(rgb_frame)
                rgb_image = rgb_frame["rgb"]
                lib.map_segment(map_handle, rgb_image.ctypes.data, rgb_image.shape[1] * 3, rgb_ts)
                segmented += 1
                end_stage("rgb_seg")
                rect_in.ready()
                if rect_in.data is not None:
                    left_rgb = np.ascontiguousarray(rect_in.data["left"])
                depth = np.ascontiguousarray(depth_frame["depth"])
                depth_raw = np.ascontiguousarray(depth_frame["depth_raw"])
                end_stage("rect_copy")
                integrate_started = time.perf_counter()
                status = lib.map_integrate_seg(map_handle, frame, pose_matrix(*robot_pose), depth, depth_raw, left_rgb,
                                               rgb_ts, np.array(vo_pose, np.float32))
                if status < 0:
                    dropped_no_rgb += 1  # the segmenter has no result for this image
                    continue
                integrate_times.append(time.perf_counter() - integrate_started)
                end_stage("integrate")
                now_perf = time.perf_counter()
                speed = 0.0
                if previous_pose:
                    speed = math.hypot(robot_pose[0] - previous_pose[0], robot_pose[1] - previous_pose[1]) / max(now_perf - previous_pose_at, 1e-3)
                previous_pose, previous_pose_at = robot_pose, now_perf
                if CFG.page_tiles:
                    lib.map_page(map_handle, robot_pose[0], robot_pose[1], robot_pose[2], speed, frame, CFG.keep_tiles, CFG.idle_frames, 1)
                end_stage("page")
                if CFG.save_debug_frames:
                    np.savez(run_dir / "frames" / f"{frame:06d}.npz", ts=depth_ts, left=left_rgb, depth=depth, depth_raw=depth_raw)
                frame += 1
                frames_since_log += 1
                end_stage("seg_pub")

            now = time.monotonic()
            end_stage("hist")
            # slam.history_generation is the 16-byte "reload the file" signal: every closure and
            # every 10 s. The file is the corrected trajectory; the rebuild skips itself if no
            # pose moved a voxel.
            generation_in.ready()
            generation_msg = generation_in.data
            generation = int(generation_msg["generation"]) if generation_msg is not None else None
            # slam starting over: a fresh map means our frame is gone; its saved map means ours holds
            slam_localized = health is not None and bool(health["localized"])
            slam_fresh = slam_localized and not bool(health["relocalized"])
            generation_reset = generation is not None and last_generation is not None and generation < last_generation
            if slam_localized and (not was_localized or (slam_fresh and not was_fresh) or generation_reset):
                if slam_fresh:
                    reset_map(lib, map_handle)
                    export = Export(lib)
                    frame = 0
                    keyframe_count = 0
                    rebuilt_generation = None
                    rebuild_started_at = None
                    with reproject_out.buf() as out:
                        out["reprojecting"] = False
                else:
                    print("[mapping] slam relocalized into its saved map again: keeping ours", flush=True)
            was_localized, was_fresh = slam_localized, slam_fresh
            if generation is not None:
                last_generation = generation
            pgo_pending = generation is not None and generation != rebuilt_generation
            if pgo_pending and frame and rebuild_started_at is None and now - last_rebuild_at >= CFG.rebuild_cooldown_s:
                history = read_slam_history(SLAM_CFG.poses_path)
                if history is not None and len(history.timestamps_ns) > 0:
                    rebuilt_generation = generation
                    last_rebuild_at = now
                    keyframe_count = lib.map_set_keyframes(map_handle, np.ascontiguousarray(history.timestamps_ns, np.int64),
                                                           np.ascontiguousarray(history.positions, np.float32),
                                                           np.ascontiguousarray(history.quaternions, np.float32),
                                                           len(history.timestamps_ns))
                    if lib.map_rebuild_pgo_start(map_handle) == 1:
                        rebuild_started_at = time.perf_counter()
                        rebuild_pgo_count = int(history.pgo_count)
                        with reproject_out.buf() as out:
                            out["reprojecting"] = True
                            out["frames_total"] = np.int32(frame)
                            out["frames_done"] = np.int32(0)
            if rebuild_started_at is not None:
                job_ms = ctypes.c_float()
                if lib.map_rebuild_poll(map_handle, ctypes.byref(job_ms)) == 1:
                    with reproject_out.buf() as out:
                        out["reprojecting"] = False
                        out["frames_total"] = np.int32(frame)
                        out["frames_done"] = np.int32(frame)
                    lib.map_counters(map_handle, counters)
                    with rebuild_out.buf() as out:
                        diff_counts = np.zeros(10, np.int32)
                        lib.map_rebuild_diff(map_handle, out["moved"], out["moved"].shape[0], out["emptied"], out["emptied"].shape[0],
                                             out["filled"], out["filled"].shape[0], diff_counts)
                        out["num_moved"], out["num_emptied"], out["num_filled"] = diff_counts[0], diff_counts[1], diff_counts[2]
                        out["count"], out["frame"] = diff_counts[3], diff_counts[4]
                    print(f"[mapping] PGO #{rebuild_pgo_count}: rebuild {frame} frames, {keyframe_count} keyframes, job {job_ms.value:.0f}ms, "
                          f"wall {1e3 * (time.perf_counter() - rebuild_started_at):.0f}ms occ={counters[0]} live={counters[1]} "
                          f"floor moved={diff_counts[0]} emptied={diff_counts[1]} filled={diff_counts[2]} | frames shifted: max {diff_counts[5] / 1000:.2f} m "
                          f"{diff_counts[7] / 10:.1f} deg at frame {diff_counts[6]}, newest 30: {diff_counts[8] / 1000:.2f} m {diff_counts[9] / 10:.1f} deg", flush=True)
                    rebuild_started_at = None

            end_stage("rebuild")
            if frame and robot_pose is not None and now - last_publish_at >= GRID2D_PERIOD_S:
                last_publish_at = now
                robot_x, robot_y, robot_yaw = robot_pose
                export.refresh(lib, map_handle)
                end_stage("export")
                # the nav grid lives in the map: changed tiles' blocks rewritten, the robot's own disc painted
                lib.map_grid(map_handle, grid2d_size, robot_x, robot_y, CFG.robot_clear_radius_m, grid2d, grid2d_origin)
                end_stage("grid2d")
                with grid2d_out.buf() as out:
                    out["grid"] = grid2d
                    out["origin"] = grid2d_origin
                    out["robot_pos"] = np.array([robot_x, robot_y], np.float32)
                    out["robot_heading"] = np.float32(robot_yaw)
                if now - last_cloud_at >= VOXELS_PERIOD_S:
                    last_cloud_at = now
                    xyz, rgb, flags, info = export.cloud()
                    outside_robot = (xyz[:, 0] - robot_x) ** 2 + (xyz[:, 1] - robot_y) ** 2 > CFG.robot_clear_radius_m ** 2
                    xyz, rgb, flags, info = xyz[outside_robot], rgb[outside_robot], flags[outside_robot], info[outside_robot]
                    voxel_count = min(len(xyz), int(CFG.max_voxels))
                    with voxels_out.buf() as out:
                        out["num_voxels"] = np.int32(voxel_count)
                        out["coords"][:voxel_count] = xyz[:voxel_count]
                        out["colors"][:voxel_count] = rgb[:voxel_count]
                        out["info"][:voxel_count] = info[:voxel_count]
                        hole_xy_parts = [holes[0] for holes in export.holes.values() if len(holes[0])]
                        hole_count = min(sum(len(part) for part in hole_xy_parts), out["holes_xy"].shape[0])
                        out["num_holes"] = np.int32(hole_count)
                        if hole_count:
                            out["holes_xy"][:hole_count] = np.concatenate(hole_xy_parts)[:hole_count]
                            out["holes_info"][:hole_count] = np.concatenate([holes[1] for holes in export.holes.values() if len(holes[1])])[:hole_count]
                        # flag bit2 marks the floor-layer voxel of a floor column: the viewer's floor label
                        out["labels"][:voxel_count] = np.where(flags[:voxel_count] & 4, -1, 1).astype(np.int8)
                        out["origin"] = grid2d_origin
                        out["robot_pos"] = np.array([robot_x, robot_y], np.float32)
                        out["robot_heading"] = np.float32(robot_yaw)
                    published_voxels = voxel_count
                end_stage("publish")

            if frame and now - last_save_at >= CFG.map_save_interval_s:
                last_save_at = now
                tiles_written, status, save_ms = save_map(lib, map_handle)
                print(f"[mapping] map saved: {tiles_written} tiles written ({TILE_COUNTS[0]} resident, {TILE_COUNTS[1]} on disk, {TILE_COUNTS[2]} still dirty), {frame} frames, {save_ms:.0f}ms" + ("" if status == 0 else " (frames.bin FAILED)"), flush=True)
                end_stage("save")

            if now - last_log_at >= CFG.log_interval_s:
                last_log_at = now
                lib.map_counters(map_handle, counters)
                integrate_ms = 1e3 * float(np.median(integrate_times)) if integrate_times else 0.0
                pose_wait_ms = 1e3 * float(np.median(pose_waits)) if pose_waits else 0.0
                print(f"[mapping] frames={frame} seg={segmented} integrate {integrate_ms:.1f}ms pose_wait {pose_wait_ms:.0f}ms voxels={published_voxels} "
                      f"occ={counters[0]} live={counters[1]} appended={counters[2]} "
                      f"dropped(no_pose={dropped_no_pose} no_mask={dropped_no_rgb} slam_lost={dropped_slam_lost}) "
                      f"keyframes={keyframe_count} pgo_pending={pgo_pending}", flush=True)
                total = sum(stage_seconds.values()) or 1.0
                print("[mapping] split ms/frame (% of wall): " + " ".join(
                    f"{name} {1e3 * seconds / max(frames_since_log, 1):.0f} ({100 * seconds / total:.0f}%)"
                    for name, seconds in stage_seconds.items()), flush=True)
                stage_seconds.clear()
                frames_since_log = 0
            time.sleep(0.002)
        if frame:
            tiles_written, status, save_ms = save_map(lib, map_handle)
            print(f"[mapping] stopping: map saved, {tiles_written} tiles written ({TILE_COUNTS[1]} on disk), {frame} frames, {save_ms:.0f}ms" + ("" if status == 0 else " (frames.bin FAILED)"), flush=True)


if __name__ == "__main__":
    main()
