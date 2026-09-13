"""Quest arm-teleop policy for the remote_session daemon.

Bridges decoded Quest controller state (quest.py) to arm commands (arm.py):
button state machine, IK solve, homing, dehome/park and slow mode. Owns the
arm ctrl and torque Writers while a session is running.

Button map, both from the Quest controllers:
  left X      toggle teleop on/off
  left Y      first press descends to park, second cuts torque
  left stick  analog modifier: left/right widens that side's gripper
  right B     tap homes; held with both triggers toggles slow mode
  grip        hold (either hand) scales that arm down: precision mode
  right A     tap toggles an episode; held 2s drops it
"""

import time

import numpy as np

from homing import park_trajectory
from quat import quat_forward_z
from quest import (HAPTIC_GAP, HAPTIC_GAP_LONG, HAPTIC_LONG, HAPTIC_SHORT,
                   PURR_THRESH)
from sound import Sound

from tracking import (
    height_remap, home_ik, gripper_command, gripper_open_pos,
    anchor_apply, capture_anchor,
    ee_anchor, to_global, new_pose_gate, pose_gate,
    constrain_to_cylinders, rate_limit, slerp_limit,
    new_precision_state, precision_reset, precision_step,
    INTERP_DURATION, HANDOFF_OFFSET, GRIPPER_CLOSED_POS_L,
    GRIPPER_CLOSED_POS_R, SLOW_MODE_DEFAULT, SLOW_MAX_SPEED,
    SLOW_MAX_ANG_SPEED, PRECISION_SCALE, PRECISION_SNAP_TAU,
)

DATASET_PREFIX = "quest_teleop"  # Same collection as the LAN app.
DT_MAX_S = 0.1                  # Clamp on the slow-mode frame delta.
DT_INIT_S = 0.02                # Assumed delta for the first slow-mode frame.
HEAD_H_MIN = 0.4                # Sanity band for latching the head plane.
HEAD_H_MAX = 2.5
PARK_TICK_S = 0.005             # Teardown park loop period.


class QuestArm:
    """Drive both arms from decoded Quest poses.

    step() is called once per loop iteration with the newest decoded state;
    it advances the button state machine and writes the arm command. All the
    machine's state lives here so the caller stays a plain relay.
    """

    def __init__(self, cfg_l, cfg_r, writers, home_duration):
        """Take the two arm configs, the four Writers, and the home ramp."""
        self.cfg_l = cfg_l
        self.cfg_r = cfg_r
        self.w_left = writers["left"]
        self.w_right = writers["right"]
        self.w_lt = writers["left_torque"]
        self.w_rt = writers["right_torque"]
        self.sound = Sound(writers.get("speaker"))
        self.home_duration = home_duration

        # Startup waypoints (native turns) for the park descent; last is home.
        self.startup_wps_l = [*cfg_l.startup_waypoints, cfg_l.home]
        self.startup_wps_r = [*cfg_r.startup_waypoints, cfg_r.home]
        self.startup_seg = cfg_l.startup_seg_durations

        self.torque_on = False
        self.teleop_active = False
        self.inited = False
        self.ref_h = None
        self.qa_left = None
        self.qa_right = None

        self.interp = False
        self.interp_t0 = 0.0
        self.interp_l = None
        self.interp_r = None

        self.homing = False
        self.home_t0 = 0.0
        self.home_l = None
        self.home_r = None

        self.dehoming = False
        self.dehome_t0 = 0.0
        self.dtraj_l = None
        self.dtraj_r = None

        # Slow mode: cylinder clamp + speed caps. cmd_* hold the rate-limiter
        # memory.
        self.slow_mode = SLOW_MODE_DEFAULT
        self.no_headset = False     # Drive on the controllers alone.
        self.anchor = None          # Frozen user frame (tracking.py).
        self.anchor_ok = False      # Last idle frame captured one.
        self.gate_l = new_pose_gate()
        self.gate_r = new_pose_gate()
        self.lpose = None           # This frame's gated poses.
        self.rpose = None
        self.cmd_pos_l = self.cmd_pos_r = None
        self.cmd_quat_l = self.cmd_quat_r = None
        self.last_slow_t = None

        # Precision clutch, per arm: hold that hand's grip to scale its motion
        # down. See precision_step() in tracking.py.
        self.prec_l = new_precision_state()
        self.prec_r = new_precision_state()

        # Episode recording. One run per daemon start, as in quest_teleop; the
        # pulses are one frame wide and the dataset daemon takes the edge.
        self.w_ds = writers.get("dataset")
        self.ep_active = False
        self.ep_toggle = False
        self.ep_drop = False
        if self.w_ds is not None:
            self.w_ds["prefix"] = DATASET_PREFIX
            self.w_ds["name"] = "%s_%s" % (DATASET_PREFIX,
                                           time.strftime("%Y%m%d_%H%M%S"))
            self.w_ds["text"] = b""     # No instruction channel yet.

        # Measured pose, refreshed by step() so the helpers can reach it.
        self._cur_l = cfg_l.home.copy()
        self._cur_r = cfg_r.home.copy()

    # --- Setup --------------------------------------------------------------
    def init_ik(self):
        """Warm up both solvers. False disables arm teleop for the session."""
        try:
            self.cfg_l.ik.init()
            self.cfg_r.ik.init()
        except Exception as e:
            print(f"  [quest] IK init failed, arm teleop disabled: {e}",
                  flush=True)
            return False
        return True

    # --- Torque -------------------------------------------------------------
    def enable_torque(self):
        """Energize both arms, flushing ctrl to the measured pose first.

        So the daemon holds where the arm is, not a stale command. Idempotent.
        """
        if self.torque_on:
            return
        self.qa_left = self.cfg_l.q2urdf(self._cur_l).copy()
        self.qa_right = self.cfg_r.q2urdf(self._cur_r).copy()
        if self.w_left is not None:
            with self.w_left.buf() as b:
                b["pos"][:] = self.cfg_l.urdf2q(self.qa_left)
                b["tau"][:] = 0.0
                b["alpha"] = 0.0
        if self.w_right is not None:
            with self.w_right.buf() as b:
                b["pos"][:] = self.cfg_r.urdf2q(self.qa_right)
                b["tau"][:] = 0.0
                b["alpha"] = 0.0
        if self.w_lt is not None:
            with self.w_lt.buf() as b:
                b["enable"] = np.ones(self.cfg_l.dof, dtype=np.bool_)
                b["tau_mode"] = np.zeros(self.cfg_l.dof, dtype=np.bool_)
                b["compliance_mode"] = False
        if self.w_rt is not None:
            with self.w_rt.buf() as b:
                b["enable"] = np.ones(self.cfg_r.dof, dtype=np.bool_)
                b["tau_mode"] = np.zeros(self.cfg_r.dof, dtype=np.bool_)
                b["compliance_mode"] = False
        self.torque_on = True
        print("  [quest] arm torque ENABLED", flush=True)

    def estop(self):
        """Cut arm torque so the arms go free. Used on Y and session end."""
        if self.w_lt is not None:
            with self.w_lt.buf() as b:
                b["enable"] = np.zeros(self.cfg_l.dof, dtype=np.bool_)
        if self.w_rt is not None:
            with self.w_rt.buf() as b:
                b["enable"] = np.zeros(self.cfg_r.dof, dtype=np.bool_)
        self.torque_on = False

    def start_home(self, cl, cr):
        """Pause teleop and interpolate both arms to home.

        Shared by the B-tap and the slow-mode toggle. Resets the IK warm start
        and the rate-limiter memory.
        """
        self.enable_torque()
        self.teleop_active = False
        self.interp = False
        self.dehoming = False
        self.home_l = cl.copy()
        self.home_r = cr.copy()
        self.home_t0 = time.monotonic()
        self.homing = True
        home_ik(self.cfg_l, self.cfg_r)
        self.cmd_pos_l = self.cmd_pos_r = None
        self.cmd_quat_l = self.cmd_quat_r = None
        precision_reset(self.prec_l)
        precision_reset(self.prec_r)

    # --- Per-frame ----------------------------------------------------------
    def precision_drift(self):
        """Each hand's carried offset (m), for the purr."""
        return (float(np.linalg.norm(self.prec_l["off"])),
                float(np.linalg.norm(self.prec_r["off"])))

    def step(self, quest, cur_left, cur_right, fresh):
        """Advance the machine and write the arm command.

        `fresh` is True only when a new packet arrived this iteration: the IK
        target advances only then, since re-solving the same pose returns
        different redundant-DOF solutions and vibrates the arm. The write at
        the end happens every frame regardless.
        """
        self._cur_l = cur_left
        self._cur_r = cur_right
        if not self.inited:
            # New quest session: arms free until a deliberate X press.
            self.estop()
            self.inited = True
        cur_l_u = self.cfg_l.q2urdf(cur_left)
        cur_r_u = self.cfg_r.q2urdf(cur_right)
        if self.qa_left is None:
            self.qa_left = cur_l_u.copy()
        if self.qa_right is None:
            self.qa_right = cur_r_u.copy()

        if fresh:
            # Gate jumps before anything consumes the pose.
            st = quest.state
            self.lpose = pose_gate(
                self.gate_l, np.asarray(st["left_pose"], dtype=np.float64),
                self.teleop_active, "L")
            self.rpose = pose_gate(
                self.gate_r, np.asarray(st["right_pose"], dtype=np.float64),
                self.teleop_active, "R")
            # No-headset: the anchor follows you and the arms until X freezes
            # it, so it is re-captured every disengaged frame.
            if self.no_headset and not self.teleop_active:
                self._capture(quest.state, cur_l_u, cur_r_u)
            self.ep_toggle = False
            self.ep_drop = False
            for ev in quest.events(self.no_headset):
                self._handle(ev, quest, cur_left, cur_right,
                             cur_l_u, cur_r_u)
            self._write_dataset()
            if self.teleop_active:
                self._track(quest.state)
            elif self.homing:
                self._advance_home()
            elif self.dehoming:
                self._advance_dehome()
            # Otherwise idle: leave qa_* frozen at the last command, NOT the
            # measured pose, which would let it sag.

        self._write()

    def _handle(self, ev, quest, cur_left, cur_right, cur_l_u, cur_r_u):
        """Run one button event from quest.py against the machine."""
        st = quest.state
        head_h = float(np.asarray(st["T_head"], dtype=np.float64)[2, 3])

        if ev == "teleop":
            self.homing = False
            if (self.no_headset and not self.teleop_active
                    and not self.anchor_ok):
                # Gripper axes too near vertical to give a yaw. Refuse.
                print("  [quest] teleop NOT enabled - point the grippers "
                      "nearer level, then press X", flush=True)
                return
            if not self.teleop_active:
                self.enable_torque()
                if self.ref_h is None and HEAD_H_MIN < head_h < HEAD_H_MAX:
                    self.ref_h = head_h             # First enable latches.
                self.interp_l = cur_l_u.copy()
                self.interp_r = cur_r_u.copy()
                self.interp_t0 = time.monotonic()
                self.interp = True
                precision_reset(self.prec_l)     # fresh 1:1 on enable
                precision_reset(self.prec_r)
                print("  [quest] teleop ENABLED (interpolating to target)",
                      flush=True)
            else:
                self.interp = False
                # Drop the rate-limiter memory so re-enabling snaps to the
                # target, not a stale command.
                self.cmd_pos_l = self.cmd_pos_r = None
                self.cmd_quat_l = self.cmd_quat_r = None
                precision_reset(self.prec_l)
                precision_reset(self.prec_r)
                print("  [quest] teleop DISABLED (holding)", flush=True)
            self.teleop_active = not self.teleop_active

        elif ev == "episode_drop":
            self.ep_drop = True
            self.ep_active = False
            quest.buzz(HAPTIC_SHORT)                # drop: 2 buzzes
            quest.buzz(HAPTIC_SHORT, HAPTIC_GAP)
            self.sound.say("Recording dropped")
            print("  [quest] episode DROPPED", flush=True)

        elif ev == "episode_toggle":
            self.ep_toggle = True
            self.ep_active = not self.ep_active
            if self.ep_active:
                quest.buzz(HAPTIC_SHORT)            # start: 1 buzz
            else:
                quest.buzz(HAPTIC_SHORT)            # stop: short | gap | long
                quest.buzz(HAPTIC_LONG, HAPTIC_GAP_LONG)
            self.sound.say("Recording started" if self.ep_active
                           else "Recording stopped")
            print("  [quest] episode %s"
                  % ("STARTED" if self.ep_active else "STOPPED"), flush=True)

        elif ev == "dehome":
            if not self.dehoming:
                self.enable_torque()                # Drive the descent.
                self.teleop_active = False
                self.interp = False
                self.homing = False
                self.dtraj_l = park_trajectory(
                    cur_left.copy(), self.startup_wps_l, self.startup_seg)
                self.dtraj_r = park_trajectory(
                    cur_right.copy(), self.startup_wps_r, self.startup_seg)
                self.dehome_t0 = time.monotonic()
                self.dehoming = True
                print("  [quest] Y: dehoming (descend to park; press Y again "
                      "to E-STOP)", flush=True)
            else:
                self.estop()
                self.dehoming = False
                self.teleop_active = False
                print("  [quest] Y again: E-STOP arm torque OFF", flush=True)

        elif ev == "slow":
            self.slow_mode = not self.slow_mode
            # Home on every switch so it isn't a jerk.
            self.start_home(cur_l_u, cur_r_u)
            quest.buzz(HAPTIC_LONG)
            self.sound.say("Tutorial mode" if self.slow_mode else "Normal mode")
            print("  [quest] SLOW MODE %s + HOME"
                  % ("ON" if self.slow_mode else "OFF"), flush=True)

        elif ev == "home":
            self.start_home(cur_l_u, cur_r_u)
            print("  [quest] HOME (B)", flush=True)

        elif ev == "height_lock":
            self.ref_h = head_h
            print("  [quest] HEIGHT PLANE LOCKED @ %.3f m" % head_h,
                  flush=True)

        elif ev == "no_headset":
            # Swapping the reference frame mid-motion would jump the arms, so
            # only while stopped.
            if self.teleop_active:
                print("  [quest] NO-HEADSET unchanged - press X to pause "
                      "teleop first", flush=True)
            else:
                self.no_headset = not self.no_headset
                self.anchor = None              # Stale in either frame.
                self.anchor_ok = False
                self.ref_h = None               # Re-latch on re-enable.
                # 4 buzzes in, 1 back out: countable without looking.
                for i in range(4 if self.no_headset else 1):
                    quest.buzz(HAPTIC_SHORT, i * HAPTIC_GAP)
                self.sound.say("No headset mode" if self.no_headset
                               else "Headset mode")
                print("  [quest] NO-HEADSET %s"
                      % ("ON" if self.no_headset else "OFF"), flush=True)

    def _capture(self, st, cur_l_u, cur_r_u):
        """Re-freeze the user frame on the live end-effectors."""
        T_head = np.asarray(st["T_head"], dtype=np.float64)
        g_l = to_global(self.lpose, T_head)
        g_r = to_global(self.rpose, T_head)
        fresh = capture_anchor(g_l, g_r, ee_anchor(self.cfg_l, cur_l_u),
                               ee_anchor(self.cfg_r, cur_r_u))
        self.anchor_ok = fresh is not None
        if fresh is not None:
            self.anchor = fresh

    def _track(self, st):
        """Solve IK for the live controller poses and blend in the ramp."""
        head_h = float(np.asarray(st["T_head"], dtype=np.float64)[2, 3])
        ref = self.ref_h if self.ref_h is not None else head_h
        lpose = self.lpose
        rpose = self.rpose
        lquat = lpose[3:]
        rquat = rpose[3:]
        lt = float(st["left_trigger"])
        rt = float(st["right_trigger"])

        if self.no_headset and self.anchor is not None:
            T_head = np.asarray(st["T_head"], dtype=np.float64)
            lpos, lquat = anchor_apply(self.anchor, to_global(lpose, T_head),
                                       "L")
            rpos, rquat = anchor_apply(self.anchor, to_global(rpose, T_head),
                                       "R")
        else:
            lpos = height_remap(lpose, ref)
            rpos = height_remap(rpose, ref)
        lpos = lpos + HANDOFF_OFFSET * quat_forward_z(lquat)
        rpos = rpos + HANDOFF_OFFSET * quat_forward_z(rquat)

        # Frame delta, shared by the precision clutch and slow mode.
        now_s = time.monotonic()
        dt_c = (min(now_s - self.last_slow_t, DT_MAX_S)
                if self.last_slow_t is not None else DT_INIT_S)
        self.last_slow_t = now_s

        # Before slow mode, as in quest_teleop: precision scales the
        # target, slow mode then caps it. teleop_active is implied here --
        # _track only runs then.
        ease = 1.0 - np.exp(-dt_c / PRECISION_SNAP_TAU)
        lpos = precision_step(self.prec_l, lpos,
                              float(st["left_squeeze"]) > PURR_THRESH,
                              PRECISION_SCALE, ease)
        rpos = precision_step(self.prec_r, rpos,
                              float(st["right_squeeze"]) > PURR_THRESH,
                              PRECISION_SCALE, ease)

        # Slow mode: clamp into this arm's half of the shell, then cap linear
        # + angular speed. Applied before IK.
        if self.slow_mode:
            lpos = constrain_to_cylinders(lpos, "left")
            rpos = constrain_to_cylinders(rpos, "right")
            self.cmd_pos_l = rate_limit(self.cmd_pos_l, lpos,
                                        SLOW_MAX_SPEED * dt_c)
            lpos = self.cmd_pos_l
            self.cmd_pos_r = rate_limit(self.cmd_pos_r, rpos,
                                        SLOW_MAX_SPEED * dt_c)
            rpos = self.cmd_pos_r
            ang = np.radians(SLOW_MAX_ANG_SPEED) * dt_c
            self.cmd_quat_l = slerp_limit(self.cmd_quat_l, lquat, ang)
            lquat = self.cmd_quat_l
            self.cmd_quat_r = slerp_limit(self.cmd_quat_r, rquat, ang)
            rquat = self.cmd_quat_r
        else:
            self.cmd_pos_l = np.asarray(lpos, dtype=np.float64).copy()
            self.cmd_pos_r = np.asarray(rpos, dtype=np.float64).copy()
            self.cmd_quat_l = np.asarray(lquat, dtype=np.float64).copy()
            self.cmd_quat_r = np.asarray(rquat, dtype=np.float64).copy()

        # Ramp from the enable pose to the live target. Blend INSIDE each
        # solve guard, so an IK miss holds the last value instead of
        # stuttering backward.
        a = (min((time.monotonic() - self.interp_t0) / INTERP_DURATION, 1.0)
             if self.interp else 1.0)
        sol_l = self.cfg_l.ik.solve(lpos.tolist(), lquat.tolist())
        if sol_l is not None and len(sol_l) > 0:
            self.qa_left[:7] = np.asarray(sol_l[:7], dtype=np.float64)
            if self.interp:
                self.qa_left = self.interp_l + a * (self.qa_left
                                                    - self.interp_l)
        sol_r = self.cfg_r.ik.solve(rpos.tolist(), rquat.tolist())
        if sol_r is not None and len(sol_r) > 0:
            self.qa_right[:7] = np.asarray(sol_r[:7], dtype=np.float64)
            if self.interp:
                self.qa_right = self.interp_r + a * (self.qa_right
                                                     - self.interp_r)
        if self.interp and a >= 1.0:
            self.interp = False
        open_l, open_r = gripper_open_pos(st["left_thumbstick"])
        gripper_command(self.cfg_l, self.qa_left, lt,
                        GRIPPER_CLOSED_POS_L, True, open_l)
        gripper_command(self.cfg_r, self.qa_right, rt,
                        GRIPPER_CLOSED_POS_R, True, open_r)

    def _advance_home(self):
        """Interpolate both arms toward home over home_duration."""
        a = min((time.monotonic() - self.home_t0) / self.home_duration, 1.0)
        home_l_u = self.cfg_l.q2urdf(
            np.asarray(self.cfg_l.home, dtype=np.float64))
        home_r_u = self.cfg_r.q2urdf(
            np.asarray(self.cfg_r.home, dtype=np.float64))
        self.qa_left = self.home_l + a * (home_l_u - self.home_l)
        self.qa_right = self.home_r + a * (home_r_u - self.home_r)
        if a >= 1.0:
            self.homing = False

    def _advance_dehome(self):
        """Sample the park trajectory; cut torque once both arms settle."""
        t_dh = time.monotonic() - self.dehome_t0
        self.qa_left = self.cfg_l.q2urdf(
            np.asarray(self.dtraj_l.at(t_dh), dtype=np.float64))
        self.qa_right = self.cfg_r.q2urdf(
            np.asarray(self.dtraj_r.at(t_dh), dtype=np.float64))
        if self.dtraj_l.done(t_dh) and self.dtraj_r.done(t_dh):
            self.estop()            # Settled at the lowest waypoint.
            self.dehoming = False
            print("  [quest] dehomed — torque OFF", flush=True)

    def _write_dataset(self):
        """One dataset.flag frame carrying this frame's episode pulses."""
        if self.w_ds is None:
            return
        with self.w_ds.buf() as b:
            b["toggle_episode"] = self.ep_toggle
            b["drop_episode"] = self.ep_drop

    def _write(self):
        """Push the current joint command to both arms, if energized."""
        if not self.torque_on:
            return
        if self.w_left is not None and self.w_left.ready():
            with self.w_left.buf() as b:
                b["pos"][:] = self.cfg_l.urdf2q(self.qa_left)
                b["tau"][:] = 0.0
                b["alpha"] = 0.0
        if self.w_right is not None and self.w_right.ready():
            with self.w_right.buf() as b:
                b["pos"][:] = self.cfg_r.urdf2q(self.qa_right)
                b["tau"][:] = 0.0
                b["alpha"] = 0.0

    def park(self, cur_left, cur_right):
        """Descend through the startup waypoints, then cut torque.

        Blocks briefly. Called at session end so the arms don't free-fall
        from a raised pose.
        """
        if not self.torque_on:
            return
        try:
            pt_l = park_trajectory(cur_left.copy(), self.startup_wps_l,
                                   self.startup_seg)
            pt_r = park_trajectory(cur_right.copy(), self.startup_wps_r,
                                   self.startup_seg)
            t0 = time.monotonic()
            while True:
                t = time.monotonic() - t0
                if self.w_left is not None:
                    with self.w_left.buf() as b:
                        b["pos"][:] = pt_l.at(t)
                        b["tau"][:] = 0.0
                        b["alpha"] = 0.0
                if self.w_right is not None:
                    with self.w_right.buf() as b:
                        b["pos"][:] = pt_r.at(t)
                        b["tau"][:] = 0.0
                        b["alpha"] = 0.0
                if pt_l.done(t) and pt_r.done(t):
                    break
                time.sleep(PARK_TICK_S)
        except Exception as e:
            print(f"  [quest] park on teardown skipped: {e}", flush=True)
        self.estop()
