"""Config and types for the mapping daemon"""
import os

import numpy as np
from bbos import Config
from bbos.registry import *

_DIR = os.path.dirname(os.path.abspath(__file__))
_depth = Config("depth")

# ============================================================================
# Config
# ============================================================================
@register
class mapping:
    library: str = os.path.join(_DIR, "libmapping.so")
    engine: str = os.path.join(_DIR, "model.engine")
    map_dir: str = os.path.join(_DIR, ".maps")                
    map_save_interval_s: float = 30.0                         
    slam_health_wait_s: float = 90.0                         
    debug_output_dir: str = os.path.join(_DIR, "logs")  # debug output dir per daemon load 
    debug_output: bool = False                                
    save_debug_frames: bool = False
    
    # slam and depth publish times differ, so we match with these constants.
    pose_gap_ms: float = 5.0                                 
    pose_wait_ms: float = 500.0
    mask_gap_ms: float = 40.0
    rebuild_cooldown_s: float = 1.0                          
    log_interval_s: float = 5.0

    voxel_size_m: float = 0.03
    max_voxels: int = 1_000_000                               # capacity of the published cloud (mapping.voxels)
    # debug topics
    max_holes: int = 200_000                                  
    max_moved: int = 100_000
    max_changed: int = 50_000
    
    grid2d_size: int = 1500                                     # side of the published nav grid in cells: 1500 x 3 cm = 45 m, origin fixed where the robot started
    robot_clear_radius_m: float = 0.30                        # the robot's own footprint is free and unpublished
    page_tiles: bool = True                                   # tiles far from the robot leave RAM (their files under map_dir are the map either way)
    keep_tiles: int = 3                                       # tiles kept resident in every direction around the robot
    idle_frames: int = 1500                                   # ... and beyond that, once untouched for this many frames

    # the segmenter runs on the left eye at half resolution and its classes are remapped onto the depth canvas
    seg_width: int = 640
    seg_height: int = 480
    canvas_width: int = int(_depth.net_width)
    canvas_height: int = int(_depth.net_height)


# ============================================================================
# Types
# ============================================================================
@realtime(ms=500)
def mapping_voxels():
    MV = mapping.max_voxels
    return [
        ("num_voxels", np.int32),
        ("coords", np.float32, (MV, 3)),
        ("colors", np.uint8, (MV, 3)),
        ("labels", np.int8, (MV,)),
        ("info", np.int32, (MV, 4)),     
        ("num_holes", np.int32),
        ("holes_xy", np.float32, (mapping.max_holes, 2)),
        ("holes_info", np.int32, (mapping.max_holes, 3)),  
        ("origin", np.float32, 2),
        ("robot_pos", np.float32, 2),
        ("robot_heading", np.float32),
    ]


@realtime(ms=200)
def mapping_grid2d():
    GS = mapping.grid2d_size
    return [
        ("grid", np.uint8, (GS, GS)), 
        ("origin", np.float32, 2),
        ("robot_pos", np.float32, 2),
        ("robot_heading", np.float32),
    ]


# -- Debug Writers ------------------------------------------------------------
@realtime(ms=200)
def mapping_rebuild():
    return [
        ("count", np.int32),        
        ("frame", np.int32),
        ("num_moved", np.int32),
        ("moved", np.float32, (mapping.max_moved, 4)),     
        ("num_emptied", np.int32),
        ("emptied", np.float32, (mapping.max_changed, 2)), 
        ("num_filled", np.int32),
        ("filled", np.float32, (mapping.max_changed, 2)),  
    ]


@state
def mapping_reproject():
    return [
        ("reprojecting", np.bool_),
        ("frames_total", np.int32),
        ("frames_done", np.int32),
    ]


