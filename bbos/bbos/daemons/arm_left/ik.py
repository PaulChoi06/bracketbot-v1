#! /usr/bin/env python3

import ctypes
import platform
import numpy as np
from pathlib import Path
from bbos.functional import curry


class Opt(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)), ("length", ctypes.c_int)]


class IKSolver(ctypes.Structure):
    pass


@curry(10)
def get_ik_solver(solver_name, urdf_content, base_link, tolerances,
                  starting_config, joint_centering_weight,
                  rik_max_iterations, centering_weights,
                  ee_link, nominal_config):
    if platform.system() != "Linux":
        return None
    return IKRust(solver_name, urdf_content, base_link, ee_link,
                  starting_config, tolerances, nominal_config,
                  joint_centering_weight, rik_max_iterations, centering_weights)


class IKRust:
    """ctypes binding for lib<solver_name>_lib.so.

    Solver tuning (mast hold, two-seed recovery, collision) lives in the Rust
    defaults, not here -- nothing in bbapps sets it at runtime.
    """
    _LIB_DIR = Path(__file__).parent

    def __init__(self, solver_name, urdf_content, base_link, ee_link,
                 starting_config, tolerances, nominal_config,
                 joint_centering_weight, rik_max_iterations, centering_weights):
        self.obj = None
        self._solver_name = solver_name
        self._urdf_content = urdf_content
        self._base_link = base_link
        self._ee_link = ee_link
        self._starting_config = starting_config
        self._nominal_config = nominal_config
        self._joint_centering_weight = joint_centering_weight
        self._rik_max_iterations = rik_max_iterations
        self._centering_weights = centering_weights
        self.tolerances = tolerances

        p_ik = ctypes.POINTER(IKSolver)
        p_d = ctypes.POINTER(ctypes.c_double)
        self.lib = ctypes.cdll.LoadLibrary(self._LIB_DIR / f'lib{solver_name}_lib.so')
        self.lib.relaxed_ik_new.restype = p_ik
        self.lib.relaxed_ik_new.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            p_d, ctypes.c_int,      # starting_config
            p_d, ctypes.c_int,      # nominal_config
            ctypes.c_double,        # joint_centering_weight
            ctypes.c_int,           # rik_max_iterations
        ]
        self.lib.reset.argtypes = [p_ik, p_d, ctypes.c_int]
        self.lib.set_tolerances.argtypes = [p_ik, p_d, ctypes.c_int]
        self.lib.solve_pose.argtypes = [p_ik, p_d, ctypes.c_int, p_d, ctypes.c_int]
        self.lib.solve_pose.restype = Opt
        self.lib.forward_kinematics.argtypes = [p_ik, p_d, ctypes.c_int]
        self.lib.forward_kinematics.restype = Opt

        # Looked up rather than bound outright: the solver libraries do not
        # share a full API, and binding a symbol the selected one lacks raises
        # at import, which takes the whole config registry down -- every
        # daemon's constants fail to load, not just this one.
        self._optional = {}
        for name in ("set_nominal_config", "set_centering_weights"):
            fn = getattr(self.lib, name, None)
            if fn is not None:
                fn.argtypes = [p_ik, p_d, ctypes.c_int]
            self._optional[name] = fn

    def _call_optional(self, name, arr, n):
        fn = self._optional.get(name)
        if fn is None:
            print(f"[ik] {self._solver_name} has no {name}(); skipping")
            return
        fn(self.obj, arr, n)

    def init(self):
        self._ensure_initialized()

    def _ensure_initialized(self):
        if self.obj is not None:
            return
        sc = self._starting_config
        sc_arr = (ctypes.c_double * len(sc))(*sc)
        if self._nominal_config is not None:
            nc_arr = (ctypes.c_double * len(self._nominal_config))(*self._nominal_config)
            nc_len = len(self._nominal_config)
        else:
            nc_arr, nc_len = ctypes.POINTER(ctypes.c_double)(), 0
        self.obj = self.lib.relaxed_ik_new(
            self._urdf_content.encode('utf-8'),
            self._base_link.encode('utf-8'),
            self._ee_link.encode('utf-8'),
            sc_arr, len(sc), nc_arr, nc_len,
            ctypes.c_double(self._joint_centering_weight),
            ctypes.c_int(self._rik_max_iterations),
        )
        if self.tolerances is not None:
            t = (ctypes.c_double * len(self.tolerances))(*self.tolerances)
            self.lib.set_tolerances(self.obj, t, len(self.tolerances))
        if self._centering_weights:
            cw = (ctypes.c_double * len(self._centering_weights))(*self._centering_weights)
            self._call_optional("set_centering_weights", cw, len(self._centering_weights))

    def reset(self, joint_state):
        self._ensure_initialized()
        js = (ctypes.c_double * len(joint_state))(*joint_state)
        self.lib.reset(self.obj, js, len(js))

    def solve(self, positions, orientations):
        self._ensure_initialized()
        pos = (ctypes.c_double * len(positions))(*positions)
        quat = (ctypes.c_double * len(orientations))(*orientations)
        xopt = self.lib.solve_pose(self.obj, pos, len(pos), quat, len(quat))
        return xopt.data[:xopt.length]

    def set_nominal(self, nominal_config):
        """Posture the centering objective pulls toward (RelaxedIK's
        init_state). reset() overwrites it, so call this after any reset()."""
        self._ensure_initialized()
        nc = (ctypes.c_double * len(nominal_config))(*nominal_config)
        self._call_optional("set_nominal_config", nc, len(nominal_config))

    def solve_with_nominal(self, positions, orientations, nominal_override):
        self._ensure_initialized()
        self.set_nominal(nominal_override)
        result = self.solve(positions, orientations)
        self.set_nominal(self._nominal_config)
        return result

    def fk(self, joint_values):
        self._ensure_initialized()
        js = (ctypes.c_double * len(joint_values))(*joint_values)
        r = self.lib.forward_kinematics(self.obj, js, len(js))
        d = r.data[:r.length]
        return np.array(d[0:3]), np.array(d[3:7])
