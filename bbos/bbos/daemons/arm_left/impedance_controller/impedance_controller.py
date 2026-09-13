from __future__ import annotations

from typing import Optional

import numpy as np
import pinocchio as pin
from bbos import Config

CFG_LEFT = Config("arm_left")
CFG_RIGHT = Config("arm_right")


def executeQuatToRotation(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64)
    assert q.shape == (4,), f"Expected quaternion shape (4,), got {q.shape}"
    n = np.linalg.norm(q)
    assert n > 1e-9, "Quaternion norm must be non-zero"
    q = q / n
    x, y, z, w = q
    r = np.empty((3, 3), dtype=np.float64)
    r[0, 0] = 1 - 2 * (y * y + z * z)
    r[0, 1] = 2 * (x * y - z * w)
    r[0, 2] = 2 * (x * z + y * w)
    r[1, 0] = 2 * (x * y + z * w)
    r[1, 1] = 1 - 2 * (x * x + z * z)
    r[1, 2] = 2 * (y * z - x * w)
    r[2, 0] = 2 * (x * z - y * w)
    r[2, 1] = 2 * (y * z + x * w)
    r[2, 2] = 1 - 2 * (x * x + y * y)
    return r


def executeResolveEeFrameId(model: pin.Model, joint_name: str) -> int:
    joint_id = model.getJointId(joint_name)
    assert 0 < joint_id < model.njoints, f"Joint {joint_name} not found in reduced model"
    def executeGetParentJoint(frame: pin.Frame) -> int:
        for key in ("parentJoint", "parent", "parentJointId"):
            if hasattr(frame, key):
                return int(getattr(frame, key))
        raise AttributeError("Unsupported pinocchio Frame parent-joint attribute")

    body_type = getattr(getattr(pin, "FrameType", object()), "BODY", None)
    body_frame_ids = [
        i
        for i, f in enumerate(model.frames)
        if executeGetParentJoint(f) == joint_id and (body_type is None or getattr(f, "type", None) == body_type)
    ]
    if len(body_frame_ids) > 0:
        return body_frame_ids[-1]
    frame_ids = [i for i, f in enumerate(model.frames) if executeGetParentJoint(f) == joint_id]
    assert len(frame_ids) > 0, f"No frame found for joint {joint_name}"
    return frame_ids[-1]


class ImpedanceController:
    def __init__(
        self,
        arm_name: str = "arm_left",
        kx_lin: Optional[float] = None,
        dx_lin: Optional[float] = None,
        kx_ang: Optional[float] = None,
        dx_ang: Optional[float] = None,
        max_f_lin: Optional[float] = None,
        max_f_ang: Optional[float] = None,
    ):
        self.cfg = Config(arm_name)
        assert self.cfg.dof >= 7, f"Expected at least 7 DOF, got {self.cfg.dof}"
        if kx_lin is None:
            kx_lin = float(self.cfg.impedance_kx_lin)
        if dx_lin is None:
            dx_lin = float(self.cfg.impedance_dx_lin)
        if kx_ang is None:
            kx_ang = float(self.cfg.impedance_kx_ang)
        if dx_ang is None:
            dx_ang = float(self.cfg.impedance_dx_ang)
        if max_f_lin is None:
            max_f_lin = float(self.cfg.impedance_max_f_lin)
        if max_f_ang is None:
            max_f_ang = float(self.cfg.impedance_max_f_ang)

        full_model = pin.buildModelFromUrdf(self.cfg.urdf_path)
        arm_ids = [full_model.getJointId(name) for name in self.cfg.joint_names[:7]]
        assert all(jid < full_model.njoints for jid in arm_ids), "Missing arm joints in URDF"
        joints_to_lock = [jid for jid in range(1, full_model.njoints) if jid not in arm_ids]
        self.model = pin.buildReducedModel(full_model, joints_to_lock, pin.neutral(full_model))
        assert self.model.nq == 7 and self.model.nv == 7, f"Expected 7-DOF model, got nq={self.model.nq}, nv={self.model.nv}"
        self.ee_frame_id = executeResolveEeFrameId(self.model, self.cfg.impedance_ee_joint)
        self.kx_lin = np.full(3, kx_lin, dtype=np.float64)
        self.dx_lin = np.full(3, dx_lin, dtype=np.float64)
        self.kx_ang = np.full(3, kx_ang, dtype=np.float64)
        self.dx_ang = np.full(3, dx_ang, dtype=np.float64)
        self.max_f_lin = float(max_f_lin)
        self.max_f_ang = float(max_f_ang)
        self.kq = np.full(self.model.nv, 0.25, dtype=np.float64)
        self.dq = np.zeros(self.model.nv, dtype=np.float64)
        self.limit_buffer_ratio = 0.08
        self.pinv_rcond = 1e-3
        self.eye = np.eye(self.model.nv, dtype=np.float64)
        self.data = self.model.createData()
        self.zero7 = np.zeros(7, dtype=np.float64)
        self.tau = np.zeros(self.cfg.dof, dtype=np.float32)
        self.vel6 = np.zeros(6, dtype=np.float64)

    def executeComputeJointLimitTorque(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        tau = -self.dq * v
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        if not np.any(finite):
            return tau
        mid = 0.5 * (lo + hi)
        half = np.maximum(0.5 * (1.0 - self.limit_buffer_ratio) * (hi - lo), 1e-4)
        z = np.clip((q - mid) / half, -0.999, 0.999)
        tau[finite] += -self.kq[finite] * (z / (1.0 - z * z + 1e-9))[finite]
        return tau

    def executeConvertMotorToUrdf(self, q_motor: np.ndarray) -> np.ndarray:
        q_motor_np = np.asarray(q_motor, dtype=np.float64)
        assert q_motor_np.shape == (self.cfg.dof,), f"Expected shape {(self.cfg.dof,)}, got {q_motor_np.shape}"
        q_urdf = np.asarray(self.cfg.q2urdf(q_motor_np.copy()), dtype=np.float64)
        assert q_urdf.shape == (self.cfg.dof,), f"Expected URDF shape {(self.cfg.dof,)}, got {q_urdf.shape}"
        return q_urdf[:7]

    def executeMapUrdfTorqueToMotor(self, tau_urdf_7: np.ndarray) -> np.ndarray:
        tau_urdf = np.asarray(tau_urdf_7, dtype=np.float64)
        assert tau_urdf.shape == (7,), f"Expected URDF torque shape (7,), got {tau_urdf.shape}"
        self.tau.fill(0.0)
        self.tau[0] = -tau_urdf[0] * self.cfg.wheel_radius #* 3 # TODO: bandaid fix for shoulder joint
        self.tau[1:7] = -tau_urdf[1:7]
        return self.tau.copy()

    def executeComputeTorque(
        self,
        q_motor: np.ndarray,
        v_motor: np.ndarray,
        x_ref: np.ndarray,
        r_ref: np.ndarray,
        gravity_scale: Optional[float] = None,
        max_tau: Optional[float] = None,
    ) -> np.ndarray:
        if gravity_scale is None:
            gravity_scale = float(self.cfg.impedance_gravity_scale)
        if max_tau is None:
            max_tau = float(self.cfg.impedance_max_tau)
        q_urdf = self.executeConvertMotorToUrdf(q_motor)
        v_urdf = self.executeConvertMotorToUrdf(v_motor)
        assert x_ref.shape == (3,), f"Expected x_ref shape (3,), got {x_ref.shape}"
        assert r_ref.shape == (3, 3), f"Expected r_ref shape (3,3), got {r_ref.shape}"

        pin.forwardKinematics(self.model, self.data, q_urdf, v_urdf)
        pin.computeJointJacobians(self.model, self.data, q_urdf)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        jac = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.vel6[:] = pin.getFrameVelocity(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector

        e_lin = x_ref - oMf.translation
        e_ang = pin.log3(r_ref @ oMf.rotation.T)
        f_lin = np.clip(self.kx_lin * e_lin - self.dx_lin * self.vel6[:3], -self.max_f_lin, self.max_f_lin)
        f_ang = np.clip(self.kx_ang * e_ang - self.dx_ang * self.vel6[3:], -self.max_f_ang, self.max_f_ang)
        tau_task = jac[:3].T @ f_lin + jac[3:].T @ f_ang

        tau_gravity = pin.rnea(self.model, self.data, q_urdf, self.zero7, self.zero7)
        tau_gravity[0] = 0.0
        print(f"tau_gravity: {tau_gravity}")
        
        tau_limit = self.executeComputeJointLimitTorque(q_urdf, v_urdf)
        null_projector = self.eye - jac.T @ np.linalg.pinv(jac.T, rcond=self.pinv_rcond)
        tau_null = null_projector @ tau_limit

        tau_urdf = float(gravity_scale) * tau_gravity + tau_task  # + tau_null
        tau_motor = self.executeMapUrdfTorqueToMotor(tau_urdf)
        tau_motor[:] = np.clip(tau_motor, -float(max_tau), float(max_tau))
        assert np.all(np.isfinite(tau_motor)), "Torque command must be finite"
        return tau_motor


class LeftImpedanceController(ImpedanceController):
    def __init__(
        self,
        kx_lin: float = CFG_LEFT.impedance_kx_lin,
        dx_lin: float = CFG_LEFT.impedance_dx_lin,
        kx_ang: float = CFG_LEFT.impedance_kx_ang,
        dx_ang: float = CFG_LEFT.impedance_dx_ang,
        max_f_lin: float = CFG_LEFT.impedance_max_f_lin,
        max_f_ang: float = CFG_LEFT.impedance_max_f_ang,
    ):
        super().__init__(
            arm_name="arm_left",
            kx_lin=kx_lin,
            dx_lin=dx_lin,
            kx_ang=kx_ang,
            dx_ang=dx_ang,
            max_f_lin=max_f_lin,
            max_f_ang=max_f_ang,
        )


class RightImpedanceController(ImpedanceController):
    def __init__(
        self,
        kx_lin: float = CFG_RIGHT.impedance_kx_lin,
        dx_lin: float = CFG_RIGHT.impedance_dx_lin,
        kx_ang: float = CFG_RIGHT.impedance_kx_ang,
        dx_ang: float = CFG_RIGHT.impedance_dx_ang,
        max_f_lin: float = CFG_RIGHT.impedance_max_f_lin,
        max_f_ang: float = CFG_RIGHT.impedance_max_f_ang,
    ):
        super().__init__(
            arm_name="arm_right",
            kx_lin=kx_lin,
            dx_lin=dx_lin,
            kx_ang=kx_ang,
            dx_ang=dx_ang,
            max_f_lin=max_f_lin,
            max_f_ang=max_f_ang,
        )
