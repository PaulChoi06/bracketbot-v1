import numpy as np
from functools import cache

UX = np.array([1,0,0])
UY = np.array([0,1,0])
UZ = np.array([0,0,1])

class Operator:
    def __init__(self, f, f_inv, _ops=None):
        self.f = f
        self.f_inv = f_inv
        self._ops = _ops or []
        self._cached_mat = None
        self._compiled_fn = None
        self._has_var = any(op[0] == 'rot_var' for op in self._ops) if self._ops else False
    
    @cache
    def __matmul__(self, other):
        fwd = lambda x: self.f(other.f(x))
        inv = lambda x: other.f_inv(self.f_inv(x))
        combined_ops = self._ops + other._ops
        return Operator(fwd, inv, combined_ops)
    
    def _ensure_compiled(self):
        """Lazily compile the operator chain for fast evaluation."""
        if self._compiled_fn is not None:
            return
        if not self._ops:
            return
        
        from math import cos, sin
        ops = self._ops
        
        # Precompute static rotations and K matrices
        static_Rs = {}
        var_Ks = {}
        var_K2s = {}
        translations = {}
        var_idx = 0
        
        for i, op in enumerate(ops):
            if op[0] == 'rot_static':
                static_Rs[i] = _rot_mat(op[1], op[2])
            elif op[0] == 'rot_var':
                K = _skew(op[1])
                var_Ks[i] = (K, var_idx)
                var_K2s[i] = K @ K
                var_idx += 1
            elif op[0] == 'trans':
                translations[i] = op[1]
        
        # Generate optimized code that writes directly to pre-allocated arrays
        lines = ["def compiled_fn(qs, out_pos, out_mat):"]
        lines.append("    DEG2RAD = 0.017453292519943295")
        
        for i in range(3):
            for j in range(3):
                lines.append(f"    R{i}{j} = {1.0 if i==j else 0.0}")
        
        lines.append("    tx, ty, tz = 0.0, 0.0, 0.0")
        
        for op_idx, op in enumerate(ops):
            if op[0] == 'rot_static':
                R_s = static_Rs[op_idx]
                for i in range(3):
                    for j in range(3):
                        terms = " + ".join([f"R{i}{m}*{R_s[m,j]}" for m in range(3)])
                        lines.append(f"    T{i}{j} = {terms}")
                for i in range(3):
                    for j in range(3):
                        lines.append(f"    R{i}{j} = T{i}{j}")
            
            elif op[0] == 'rot_var':
                K, var_idx = var_Ks[op_idx]
                K2 = var_K2s[op_idx]
                lines.append(f"    th = qs[{var_idx}] * DEG2RAD")
                lines.append(f"    c, s = _cos(th), _sin(th)")
                for i in range(3):
                    for j in range(3):
                        eye = 1.0 if i==j else 0.0
                        lines.append(f"    Rj{i}{j} = {eye} + s*{K[i,j]} + (1-c)*{K2[i,j]}")
                for i in range(3):
                    for j in range(3):
                        terms = " + ".join([f"R{i}{m}*Rj{m}{j}" for m in range(3)])
                        lines.append(f"    T{i}{j} = {terms}")
                for i in range(3):
                    for j in range(3):
                        lines.append(f"    R{i}{j} = T{i}{j}")
            
            elif op[0] == 'trans':
                t = translations[op_idx]
                lines.append(f"    tx += R00*{t[0]} + R01*{t[1]} + R02*{t[2]}")
                lines.append(f"    ty += R10*{t[0]} + R11*{t[1]} + R12*{t[2]}")
                lines.append(f"    tz += R20*{t[0]} + R21*{t[1]} + R22*{t[2]}")
        
        # Write to pre-allocated arrays
        lines.append("    out_pos[0], out_pos[1], out_pos[2] = tx, ty, tz")
        for i in range(3):
            for j in range(3):
                lines.append(f"    out_mat[{i},{j}] = R{i}{j}")
        lines.append("    out_mat[0,3], out_mat[1,3], out_mat[2,3] = tx, ty, tz")
        
        code = "\n".join(lines)
        local_ns = {"_cos": cos, "_sin": sin}
        exec(code, local_ns)
        self._compiled_fn = local_ns["compiled_fn"]
        self._out_pos = np.zeros(3, dtype=np.float64)
        self._out_mat = np.eye(4, dtype=np.float64)
    
    def _compute_mat(self, qs=None):
        """Compute 4x4 matrix using compiled function if available."""
        if not self._ops:
            # Fallback to function-based computation
            e1 = self.f(np.array([1., 0., 0.]))
            e2 = self.f(np.array([0., 1., 0.]))
            e3 = self.f(np.array([0., 0., 1.]))
            origin = self.f(np.array([0., 0., 0.]))
            R = np.column_stack([e1 - origin, e2 - origin, e3 - origin])
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = origin
            return T
        
        self._ensure_compiled()
        qs_arr = qs if qs is not None else []
        self._compiled_fn(qs_arr, self._out_pos, self._out_mat)
        return self._out_mat
    
    def mat(self, qs=None):
        """Return the 4x4 homogeneous transformation matrix."""
        if self._has_var:
            if qs is None:
                raise ValueError("Variable rotation requires qs parameter")
            return self._compute_mat(qs)
        if self._cached_mat is None:
            self._cached_mat = self._compute_mat()
        return self._cached_mat.copy()
    
    def __call__(self, x, qs=None):
        """Transform point(s) x. Pass qs for variable rotations."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.squeeze()
        
        M = self.mat(qs) if self._has_var else self.mat()
        R, t = M[:3, :3], M[:3, 3]
        
        if x.ndim == 1:
            return R @ x + t
        else:
            return (R @ x.T).T + t
    
    def inv(self):
        return Operator(self.f_inv, self.f)
    
    def xytheta(self, qs=None): 
        xyth = self.pos(qs)
        xyth[2] = self.rpy(qs)[2]
        return xyth
    
    def quat(self, qs=None):
        """Extract the rotation quaternion from the transformation."""
        M = self.mat(qs) if self._has_var else self.mat()
        R = M[:3, :3]
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])
    
    def pos(self, qs=None, copy=True):
        """Get the final position after applying transformation to origin."""
        if self._has_var and self._ops:
            self._ensure_compiled()
            self._compiled_fn(qs, self._out_pos, self._out_mat)
            return self._out_pos.copy() if copy else self._out_pos
        M = self.mat()
        return M[:3, 3].copy()
    
    def rpy(self, qs=None):
        """Extract roll, pitch, yaw angles (in radians) from the transformation."""
        M = self.mat(qs) if self._has_var else self.mat()
        R = M[:3, :3]
        
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw])

def trans(t):
    """Translation operator."""
    t = np.asarray(t, dtype=float)
    def f(x):
        return x + t
    def f_inv(x):
        return x - t
    return Operator(f, f_inv, [('trans', t.copy())])

def rot(axis, angle):
    """
    Rotation operator around `axis`.
    Right-hand rule: positive angle = CCW when looking along +axis.
    
    Args:
        axis: rotation axis (3D vector)
        angle: angle in degrees, or None for variable rotation (angle from qs)
    
    Example:
        rot(UZ, 45)              # static 45 degree rotation
        rot(UZ, None)            # variable rotation, angle from qs (in order)
        T = rot(UZ, None) @ trans([1,0,0]) @ rot(UX, None)
        T.pos(qs=[45., 30.])     # first variable rot gets 45, second gets 30
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    
    if angle is not None:
        # Static rotation
        θ = np.deg2rad(angle)
        c, s = np.cos(θ), np.sin(θ)
        
        def f(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return x*c + np.cross(a, x)*s + a*np.dot(a, x)*(1-c)
            elif x.ndim == 2 and x.shape[1] == 3:
                return x*c + np.cross(np.broadcast_to(a, x.shape), x)*s + np.outer(np.dot(x, a), a)*(1-c)
            else:
                raise ValueError("Input must be shape (3,) or (N,3)")
        
        def f_inv(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return x*c - np.cross(a, x)*s + a*np.dot(a, x)*(1-c)
            elif x.ndim == 2 and x.shape[1] == 3:
                return x*c - np.cross(np.broadcast_to(a, x.shape), x)*s + np.outer(np.dot(x, a), a)*(1-c)
            else:
                raise ValueError("Input must be shape (3,) or (N,3)")
        
        return Operator(f, f_inv, [('rot_static', a.copy(), float(angle))])
    
    else:
        # Variable rotation - index assigned during compilation based on order
        def f(x):
            return x  # placeholder
        def f_inv(x):
            return x  # placeholder
        
        return Operator(f, f_inv, [('rot_var', a.copy())])

def quat(q):
    """
    Create a rotation operator from a quaternion [x, y, z, w].
    Uses efficient direct quaternion rotation formula.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Quaternion must be shape (4,)")
    
    x, y, z, w = q
    
    # Normalize the quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0:
        print("Cannot normalize zero quaternion", flush=True)
        x, y, z, w = 0, 0, 0, 1
        norm = 1
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    def f(v):
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            # Single vector - use optimized quaternion rotation formula
            # v' = v + 2*w*(q_vec × v) + 2*(q_vec × (q_vec × v))
            # where q_vec = [x, y, z]
            qv_cross = np.array([
                y*v[2] - z*v[1],
                z*v[0] - x*v[2],
                x*v[1] - y*v[0]
            ])
            qv_cross_cross = np.array([
                y*qv_cross[2] - z*qv_cross[1],
                z*qv_cross[0] - x*qv_cross[2],
                x*qv_cross[1] - y*qv_cross[0]
            ])
            return v + 2*w*qv_cross + 2*qv_cross_cross
        elif v.ndim == 2 and v.shape[1] == 3:
            # Batch of vectors
            qv_cross = np.empty_like(v)
            qv_cross[:, 0] = y*v[:, 2] - z*v[:, 1]
            qv_cross[:, 1] = z*v[:, 0] - x*v[:, 2]
            qv_cross[:, 2] = x*v[:, 1] - y*v[:, 0]
            
            qv_cross_cross = np.empty_like(v)
            qv_cross_cross[:, 0] = y*qv_cross[:, 2] - z*qv_cross[:, 1]
            qv_cross_cross[:, 1] = z*qv_cross[:, 0] - x*qv_cross[:, 2]
            qv_cross_cross[:, 2] = x*qv_cross[:, 1] - y*qv_cross[:, 0]
            
            return v + 2*w*qv_cross + 2*qv_cross_cross
        else:
            raise ValueError("Input must be shape (3,) or (N,3)")
    
    def f_inv(v):
        # Inverse rotation is rotation by conjugate quaternion [-x, -y, -z, w]
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            # Single vector
            qv_cross = np.array([
                -y*v[2] + z*v[1],
                -z*v[0] + x*v[2],
                -x*v[1] + y*v[0]
            ])
            qv_cross_cross = np.array([
                -y*qv_cross[2] + z*qv_cross[1],
                -z*qv_cross[0] + x*qv_cross[2],
                -x*qv_cross[1] + y*qv_cross[0]
            ])
            return v + 2*w*qv_cross + 2*qv_cross_cross
        elif v.ndim == 2 and v.shape[1] == 3:
            # Batch of vectors
            qv_cross = np.empty_like(v)
            qv_cross[:, 0] = -y*v[:, 2] + z*v[:, 1]
            qv_cross[:, 1] = -z*v[:, 0] + x*v[:, 2]
            qv_cross[:, 2] = -x*v[:, 1] + y*v[:, 0]
            
            qv_cross_cross = np.empty_like(v)
            qv_cross_cross[:, 0] = -y*qv_cross[:, 2] + z*qv_cross[:, 1]
            qv_cross_cross[:, 1] = -z*qv_cross[:, 0] + x*qv_cross[:, 2]
            qv_cross_cross[:, 2] = -x*qv_cross[:, 1] + y*qv_cross[:, 0]
            
            return v + 2*w*qv_cross + 2*qv_cross_cross
        else:
            raise ValueError("Input must be shape (3,) or (N,3)")
    
    return Operator(f, f_inv)

def rmat(R):
    """
    Rotation operator from a 3x3 rotation matrix. from https://github.com/UT-Austin-RPL/deoxys_control/blob/main/deoxys/deoxys/utils/transform_utils.py
    """
    M = np.asarray(R, dtype=np.float32)[:3, :3]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return quat(q1[inds])

def rpy(rpy):
    """
    from https://github.com/UT-Austin-RPL/deoxys_control/blob/main/deoxys/deoxys/utils/transform_utils.py
    """
    euler = np.asarray(rpy, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shaped euler {euler}"
    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci

    return rmat(mat)

def rmat_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])

def pose_to_matrix(pos, quat):
    """Convert position + quaternion (xyzw) to 4x4 transform matrix"""
    x, y, z, w = quat
    T = np.eye(4)
    T[0, 0] = 1 - 2*(y*y + z*z)
    T[0, 1] = 2*(x*y - z*w)
    T[0, 2] = 2*(x*z + y*w)
    T[1, 0] = 2*(x*y + z*w)
    T[1, 1] = 1 - 2*(x*x + z*z)
    T[1, 2] = 2*(y*z - x*w)
    T[2, 0] = 2*(x*z - y*w)
    T[2, 1] = 2*(y*z + x*w)
    T[2, 2] = 1 - 2*(x*x + y*y)
    T[:3, 3] = pos
    return T

def _skew(a):
    """Skew-symmetric matrix for cross product: skew(a) @ b = a × b"""
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)

def _rot_mat(axis, angle_deg):
    """Fast rotation matrix via Rodrigues formula"""
    a = axis / np.linalg.norm(axis)
    th = angle_deg * 0.017453292519943295  # pi/180
    c, s = np.cos(th), np.sin(th)
    K = _skew(a)
    return np.eye(3) + s * K + (1 - c) * (K @ K)


if __name__ == "__main__":
    import timeit
    
    print("=== tf.py tests ===\n")
    
    # Test basic Operator API
    print("1. Testing Operator API...")
    T1 = trans([1, 0, 0]) @ rot(UZ, 90)
    p = T1([0, 0, 0])
    assert np.allclose(p, [1, 0, 0]), f"trans+rot failed: {p}"
    
    T2 = rot(UZ, 90) @ trans([1, 0, 0])
    p2 = T2([0, 0, 0])
    assert np.allclose(p2, [0, 1, 0]), f"rot+trans failed: {p2}"
    print("   Operator API: PASS")
    
    # Test mat() extraction
    print("2. Testing mat() extraction...")
    T = trans([1, 2, 3]) @ rot(UX, 45)
    M = T.mat()
    assert M.shape == (4, 4), f"mat shape wrong: {M.shape}"
    assert np.allclose(M[:3, 3], [1, 2, 3]), f"translation wrong: {M[:3, 3]}"
    print("   mat(): PASS")
    
    # Test _rot_mat matches rot()
    print("3. Testing _rot_mat vs rot()...")
    for axis, angle in [(UX, 45), (UY, -30), (UZ, 90), ([1,1,1], 60)]:
        R_fast = _rot_mat(np.array(axis), angle)
        R_slow = rot(axis, angle).mat()[:3, :3]
        assert np.allclose(R_fast, R_slow, atol=1e-10), f"mismatch for axis={axis}, angle={angle}"
    print("   _rot_mat: PASS")
    
    # Test variable rotations with qs parameter
    print("4. Testing variable rotations...")
    
    # Simple test: rotate then translate
    T = rot(UZ, None) @ trans([1, 0, 0])
    
    # At 0 degrees, should be at [1, 0, 0]
    pos = T.pos(qs=[0.])
    assert np.allclose(pos, [1, 0, 0]), f"var rot 0deg: {pos}"
    
    # At 90 degrees, should be at [0, 1, 0]
    pos = T.pos(qs=[90.])
    assert np.allclose(pos, [0, 1, 0], atol=1e-10), f"var rot 90deg: {pos}"
    
    # Test mat() with qs
    M = T.mat(qs=[45.])
    assert M.shape == (4, 4), f"mat(qs) shape wrong"
    
    # Test matches static Operator
    T_var = rot(-UY, None) @ trans([0, 0, -0.035]) @ rot(-UX, None) @ trans([0, 0, -0.111])
    T_static = rot(-UY, 30) @ trans([0, 0, -0.035]) @ rot(-UX, 45) @ trans([0, 0, -0.111])
    
    pos_var = T_var.pos(qs=[30., 45.])
    pos_static = T_static.pos()
    assert np.allclose(pos_var, pos_static, atol=1e-10), f"var vs static: {pos_var} vs {pos_static}"
    
    print("   Variable rotations: PASS")
    
    # Test caching for static chains
    print("5. Testing matrix caching...")
    T_cached = rot(UZ, 45) @ trans([1, 0, 0]) @ rot(UX, 30)
    M1 = T_cached.mat()
    M2 = T_cached.mat()
    assert M1 is not M2, "mat() should return copy"
    assert np.allclose(M1, M2), "cached matrices should be equal"
    print("   Matrix caching: PASS")
    
    # Benchmark
    print("\n6. Benchmarking...")
    
    # Build 6-joint chain with variable rotations
    qvs = [-UY, -UX, -UY, UZ, UX, -UY]
    offs = [[0,0,0], [0,0,-0.035], [0,0,-0.111], [0,0,-0.051], [0,0,-0.098], [0,0,-0.0648]]
    
    T_chain = rot(qvs[0], None) @ trans(offs[0])
    for i in range(1, 6):
        T_chain = T_chain @ rot(qvs[i], None) @ trans(offs[i])
    
    # Static chain for comparison
    T_static_chain = rot(qvs[0], 45) @ trans(offs[0])
    for i in range(1, 6):
        T_static_chain = T_static_chain @ rot(qvs[i], 30) @ trans(offs[i])
    
    qs_test = [45., 30., 60., 10., 20., 15.]
    
    # Warmup
    for _ in range(100):
        T_chain.pos(qs=qs_test)
        T_chain.pos(qs=qs_test, copy=False)
        T_static_chain.pos()
    
    n_runs = 1000
    t_var = timeit.timeit(lambda: T_chain.pos(qs=qs_test), number=n_runs)
    t_var_nocopy = timeit.timeit(lambda: T_chain.pos(qs=qs_test, copy=False), number=n_runs)
    t_static = timeit.timeit(lambda: T_static_chain.pos(), number=n_runs)
    
    print(f"   T.pos(qs=...) (variable):      {(t_var / n_runs) * 1e6:.2f} μs")
    print(f"   T.pos(qs=..., copy=False):     {(t_var_nocopy / n_runs) * 1e6:.2f} μs")
    print(f"   T.pos() (static/cached):       {(t_static / n_runs) * 1e6:.2f} μs")
    
    print("\n=== All tests passed ===")