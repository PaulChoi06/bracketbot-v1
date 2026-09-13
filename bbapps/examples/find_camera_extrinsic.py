# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "trimesh<5.0.0",
#   "yourdfpy>=0.0.56",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""
Camera -> arm_base extrinsic, via URDF forward kinematics.

Read the whole message this prints before trusting the numbers: unlike the
wrist camera (l_wrist_cam__wrist_cam / wrist_cam__wrist_cam -- a link placed
deliberately at the lens), this robot's URDF has NO dedicated optical frame
for the head stereo camera. The only nearby links are structural/cover
parts from the CAD export (head__head, camera_cover__camera_cover) that
happen to be in the neighborhood, connected via a chain of ~zero-offset
fixed joints -- i.e. this script gives you a STARTING approximation of
where the camera roughly is, not a calibrated optical frame. Good enough to
get a pick-and-place pipeline running end to end; verify empirically (see
bottom of this file's output) before trusting it for anything precise.

Confirmed directly in the URDF before writing this: arm_right's Config
literally reuses arm_left's exact urdf_path (same file), and both arms'
mast joints (rj0, lj0) parent to the SAME "arm_base" link -- so there is
exactly one arm_base frame, shared by both arms, and this script's output
applies to arm_left and arm_right equally. "arm_base" is also exactly the
frame cfg.ik.solve()/.fk() already operate in for both arms, so composing
a detected tag's camera-frame pose with this transform lands directly in
IK's own coordinate frame -- no extra conversion needed.

Run:    uv run examples/find_camera_extrinsic.py
Saves:  /home/bracketbot/camera_extrinsic.json
"""
import json
import numpy as np
import yourdfpy
from bbos import Config
from bbos.tf import rmat_to_quat

CANDIDATES = ["head__head", "camera_cover__camera_cover"]
REFERENCE = ["left_eef", "right_eef"]


def describe(label, T):
    pos = T[:3, 3]
    quat_xyzw = rmat_to_quat(T[:3, :3])
    print(f"\n{label}  (relative to arm_base)")
    print(f"  xyz (m):     {np.round(pos, 4).tolist()}")
    print(f"  quat (xyzw): {np.round(quat_xyzw, 4).tolist()}")
    return {"xyz": pos.tolist(), "quat_xyzw": quat_xyzw.tolist()}


def main():
    cfg = Config("arm_left")  # arm_right uses this exact same URDF file -- confirmed
    urdf = yourdfpy.URDF.load(cfg.urdf_path, load_meshes=False,
                              build_scene_graph=True, build_collision_scene_graph=False)
    urdf.update_cfg({j: 0.0 for j in urdf.joint_map.keys()})

    def transform(frame_to):
        return urdf.get_transform(frame_to=frame_to, frame_from="arm_base")

    print(f"URDF: {cfg.urdf_path}")
    print("(arm_right shares this exact file; one arm_base, shared by both arms)")

    out = {"urdf_path": cfg.urdf_path, "candidates": {}, "reference": {}}

    for name in CANDIDATES:
        try:
            out["candidates"][name] = describe(f"[CANDIDATE] {name}", transform(name))
        except Exception as e:
            print(f"\n[CANDIDATE] {name}: FAILED ({e})")

    for name in REFERENCE:
        try:
            out["reference"][name] = describe(f"[reference] {name}", transform(name))
        except Exception as e:
            print(f"\n[reference] {name}: FAILED ({e})")

    print("\nOther link names containing 'head'/'cam'/'cover' (in case neither "
          "candidate above turns out right):")
    for name in urdf.link_map.keys():
        if any(s in name.lower() for s in ("head", "cam", "cover")):
            print(" ", name)

    out_path = "/home/bracketbot/camera_extrinsic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")

    print("""
NEXT STEP -- sanity-check before trusting either candidate (5 minutes):
  1. Put an ArUco tag somewhere the head camera can see AND the arm can
     reach (e.g. flat on the tray/table in front of the robot).
  2. Run examples/view_pnp.py pointed at that tag's id; note its reported
     camera-frame pose (position, at least).
  3. Jog/teleop the gripper to just touch the tag's center (or hover
     directly above it at a measured height), then read the arm's OWN
     idea of where it is: uv run examples/describe_pose.py --arm arm_left
     --motor '[...]' with its current motor positions -- gives EE position
     in arm_base frame directly from cfg.ik.fk().
  4. Compose: candidate_extrinsic @ tag_pose_in_camera_frame, and compare
     against the arm-reported position from step 3. Off by a few cm at
     typical working distance is normal to true-up with a hand-tuned
     correction offset; off by tens of cm or with flipped signs means try
     the other candidate link, or fall back to the depth daemon's
     T_base_cam constants (bbos/daemons/depth/constants.py) as a rougher
     starting point, or do a proper multi-point calibration using the arm
     itself as the measurement tool (touch 3+ known points, solve the
     rigid transform) if precision matters more than the time it costs.
""")


if __name__ == "__main__":
    main()
