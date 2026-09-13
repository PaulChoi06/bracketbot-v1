from bbos import Reader, Writer, Config, Type
import bbos.tf as tf
import numpy as np
from datetime import datetime
from pathlib import Path

def curry(num_args):
    def decorator(fn):
        def init(*args, **kwargs):
            def call(*more_args, **more_kwargs):
                all_args = args + more_args
                all_kwargs = dict(**kwargs, **more_kwargs)
                if len(all_args) + len(all_kwargs) >= num_args:
                    return fn(*all_args, **all_kwargs)
                else:
                    return init(*all_args, **all_kwargs)
            return call
        return init()
    return decorator

@curry(4)
def getGravComp(qvs, ms, fk_ops, qs):
    """fk_ops is list of pre-built operators with variable rotations."""
    n = len(ms)
    g = 9.81
    
    # Get positions - this part uses tf.py compiled code
    pqs = np.array([T.pos(qs=qs) for T in fk_ops])
    
    # Simplified: cmps=0, so vcmps=0, so mpvs[i,j] = pqs[j] - pqs[i]
    # Vectorized gravity comp
    torques = np.zeros(n)
    for i in range(n):
        qv = qvs[i]
        qv_norm_sq = qv[0]**2 + qv[1]**2 + qv[2]**2
        for j in range(i, n):
            if ms[j] == 0:
                continue
            # d = pqs[j] - pqs[i]
            dx = pqs[j, 0] - pqs[i, 0]
            dy = pqs[j, 1] - pqs[i, 1]
            dz = pqs[j, 2] - pqs[i, 2]
            # project perpendicular to axis
            dot = dx*qv[0] + dy*qv[1] + dz*qv[2]
            scale = dot / qv_norm_sq
            px = dx - scale * qv[0]
            py = dy - scale * qv[1]
            # cross([0,0,-m*g], [px,py,pz]) = [-m*g*py, m*g*px, 0]
            m_g = ms[j] * g
            torques[i] += (m_g**2 * (px**2 + py**2)) ** 0.5
    return torques

qvs = np.array([-tf.UY, -tf.UX, -tf.UY, tf.UZ, tf.UX, -tf.UY])

@curry(2)
def ARM_FK(tilt_angle, qvs):
    """Build FK operators once with variable rotations. Returns list of 6 operators."""
    offs = np.array([[0,0,0], [0,0,-0.035], [0,0,-0.111], [0,0,-0.051], [0,0,-0.098], [0,0,-0.0648]])
    sh_tilt_side = tf.rot(tf.UZ, tilt_angle)
    sh_tilt_up = tf.rot(tf.UX, tilt_angle)
    sh_tilt = sh_tilt_up @ sh_tilt_side
    tilt_axis = sh_tilt_side.inv()(tf.UX)
    T1 = sh_tilt @ tf.trans(offs[0]) @ tf.rot(qvs[0], None)
    T2 = T1 @ tf.trans(offs[1]) @ tf.rot(qvs[1], None)
    T3 = T2 @ tf.rot(tilt_axis, -tilt_angle) @ tf.trans(offs[2]) @ tf.rot(qvs[2], None)
    T4 = T3 @ tf.trans(offs[3]) @ tf.rot(qvs[3], None)
    T5 = T4 @ tf.trans(offs[4]) @ tf.rot(qvs[4], None)
    T6 = T5 @ tf.trans(offs[5]) @ tf.rot(qvs[5], None)
    return [T1, T2, T3, T4, T5, T6]

qvs = np.array([-tf.UY, -tf.UX, -tf.UY, tf.UZ, tf.UX, -tf.UY])
ms = np.array([0,0.166,  0.166,  0.1075, 0.1075, 0.1075])
fk_ops = ARM_FK(20, qvs)  # build operators once with no tilt
gc_fn = getGravComp(qvs, ms, fk_ops)  # curry with fk_ops, qs passed later


CFG = Config("arm_left")
dof = CFG.dof
log_dir = Path(".data") / f"calibrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir.mkdir(parents=True, exist_ok=True)

n = 5000
dat = np.zeros((n,), dtype=[
    ('t', 'f8'),
    ('pd', 'f8', (dof,)),
    ('vd', 'f8', (dof,)),
    ('ps', 'f8', (dof,)),
    ('vs', 'f8', (dof,)),
    ('ts', 'f8', (dof,)),
])
i = 0
t0 = None
pd = np.zeros(dof)
gc = np.zeros(dof)
vd = np.zeros(dof)
ps = np.zeros(dof)
dev = True
with Writer("arm_left.torque", Type("arm_left_torque")) as w_torque:
    w_torque['enable'] = np.ones(dof, dtype=np.bool_)
    with Writer("arm_left.ctrl", Type("arm_left_ctrl")) as w_ctrl, Reader("arm_left.state", sync=True) as r_state:
        print(f"Running calibration trajectory ({n} samples)...", flush=True)
        while i < n:
            if r_state.ready():
                if t0 is None:
                    t0 = r_state.data['timestamp']
                ps = r_state.data['pos']
                gc[1:-1] = gc_fn(ps[1:-1] * 360)
                gc[1:-1] = 1/(np.array(CFG.Kp[1:-1])/10) * 1/(np.array(CFG.kt[1:-1])) * gc[1:-1]
                dat[i]['t'] = (r_state.data['timestamp'] - t0) / np.timedelta64(1, 's')
                dat[i]['pd'] = gc 
                dat[i]['vd'] = vd
                dat[i]['ps'] = r_state.data['pos']
                dat[i]['vs'] = r_state.data['vel']
                dat[i]['ts'] = r_state.data['torque']
            w_ctrl['pos'] = ps
            if w_ctrl.ready():
                i += 1
    w_torque['enable'] = np.zeros(dof, dtype=np.bool_)
if i < n:
    print(f"Warning: only {i} samples were collected, not saving.")
else:
    if dev:
        dat.tofile(Path(".data") / "data.dat")
    else:
        dat.tofile(log_dir / "data.dat")
    print(f"Saved calibration data to {log_dir / 'data.dat'}")
