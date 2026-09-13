"""Bridge baseboard commands and status to BBOS drive, IMU, and health topics.

The baseboard owns the 200 Hz balance loop; this host daemon handles I/O only.
"""
import math
import time

from bbos import Config, Reader, Type, Writer

from driver import Link, build_config, clamp, describe_config

CFG = Config("base")

MODE_BALANCE = CFG.MODE_BALANCE
MODE_LEAN = CFG.MODE_LEAN
MODE_TWIST = CFG.MODE_TWIST

MODE_NAMES = {
    MODE_BALANCE: "BALANCE",
    MODE_LEAN: "LEAN",
    MODE_TWIST: "TWIST (velocity, NOT balancing)",
}

if __name__ == "__main__":
    link = Link(CFG.port, CFG.baud, CFG.expected_link_version)
    period = 1.0 / CFG.command_hz
    next_send = time.monotonic()
    next_health = time.monotonic() + 1.0
    next_status = time.monotonic() + 10.0
    status_count = 0
    mode = None
    prev_mode = None
    lean_deg = clamp(float(CFG.lean_angle_deg),
                     CFG.lean_angle_min_deg, CFG.lean_angle_max_deg)
    mode_update = None
    mode_update_packets = 0
    mode_deadline = time.monotonic() + CFG.mode_command_timeout_s
    drive_deadline = time.monotonic()
    odrive_settings = Config("odrive").settings
    left_axis = int(CFG.left_axis)
    applied_gains = (
        float(odrive_settings[
            f"axis{left_axis}.controller.config.vel_gain"]),
        float(odrive_settings[
            f"axis{left_axis}.controller.config.vel_integrator_gain"]),
    )
    last_gains = None
    ff_tau_s_nm = float(CFG.ff_tau_s_nm)
    ff_tau_k_nm = float(CFG.ff_tau_k_nm)
    ff_yaw_nm = float(CFG.ff_yaw_nm)
    last_ff_tune = (ff_tau_s_nm, ff_tau_k_nm, ff_yaw_nm)
    profile_enabled = bool(CFG.twist_profile_enabled)
    profile_amax = float(CFG.twist_profile_amax)
    profile_jmax_acc = float(CFG.twist_profile_jmax_acc)
    profile_jmax_dec = float(CFG.twist_profile_jmax_dec)
    last_profile_tune = (profile_enabled, profile_amax,
                         profile_jmax_acc, profile_jmax_dec)
    twist_tune_phase = 0

    # ODrive latches Iq_measured after IDLE, so zero it after command inactivity.
    IQ_ZERO_AFTER_S = 0.5
    iq_cmd_active = 0.0

    LEAN_STUCK_S = 1.0            # asserted-but-inactive before we act
    LEAN_CLEAR_S = 0.2            # how long to drop the request (> 5 ticks)
    LEAN_COOLDOWN_S = 5.0         # minimum spacing between attempts

    LEAN_SAFE_PITCH_RAD = 0.175
    lean_stuck_since = None
    lean_clear_until = 0.0
    lean_next_attempt = 0.0
    lean_relatch_count = 0
    reset_cmd_integral = False
    last_pitch = 0.0
    last_imu_diag_ms = None
    print(f"[base] link up on {CFG.port}", flush=True)

    try:
        with Reader('drive.ctrl') as r_ctrl, \
             Reader('base.mode') as r_mode, \
             Reader('base.cmd_int_reset') as r_cmd_int_reset, \
             Reader('drive.tune') as r_tune, \
             Writer('drive.state', Type("drive_state")) as w_state, \
             Writer('drive.status', Type("drive_status"),
                    keeptime=False) as w_status, \
             Writer('imu.orientation', Type("imu_orientation")) as w_orient, \
             Writer('imu.raw', Type("imu_raw")) as w_raw, \
             Writer('imu.diagnostics', Type('imu_diagnostics'), keeptime=False) as w_imu_diag, \
             Writer('base.health', Type("base_health"),
                    keeptime=False) as w_health:

            base_config = build_config(CFG)
            link.set_default_mode(CFG.default_mode)
            print("[base] default mode persisted: %s" %
                  MODE_NAMES.get(CFG.default_mode, CFG.default_mode), flush=True)
            print("[base] host mapping loaded (not pushed): "
                  + describe_config(base_config), flush=True)

            prev_live = False
            v = w = 0.0
            while True:
                now = time.monotonic()

                if r_tune.ready():
                    d = r_tune.data
                    vg = float(d['vel_gain'])
                    vig = float(d['vel_integrator_gain'])
                    # NaN would be written straight through to the drives.
                    if (math.isfinite(vg) and math.isfinite(vig) and
                            (vg, vig) != last_gains):
                        for node in (CFG.left_axis, CFG.right_axis):
                            link.odrive_set_vel_gains(node, vg, vig)
                        applied_gains = (vg, vig)
                        last_gains = (vg, vig)
                        print("[base] gains -> vel_gain=%.3f vel_integrator_gain=%.3f"
                              % (vg, vig), flush=True)
                    next_ff = (float(d['ff_tau_s_nm']),
                               float(d['ff_tau_k_nm']),
                               float(d['ff_yaw_nm']))
                    if next_ff != last_ff_tune:
                        ff_tau_s_nm, ff_tau_k_nm, ff_yaw_nm = next_ff
                        last_ff_tune = next_ff
                        print("[base] twist feedforward -> tau_s=%.3fNm "
                              "tau_k=%.3fNm yaw=%.3fNm"
                              % next_ff, flush=True)
                    next_profile = (
                        float(d['prof_enabled']) >= 0.5,
                        float(d['prof_amax']),
                        float(d['prof_jmax_acc']),
                        float(d['prof_jmax_dec']))
                    if next_profile != last_profile_tune:
                        (profile_enabled, profile_amax,
                         profile_jmax_acc, profile_jmax_dec) = next_profile
                        last_profile_tune = next_profile
                        print("[base] twist profile -> enabled=%s "
                              "amax=%.2fm/s^2 jerk=%.2f/%.2fm/s^3"
                              % next_profile, flush=True)

                if r_ctrl.ready():
                    twist = r_ctrl.data['twist']
                    requested_v = float(twist[0])
                    requested_w = float(twist[1])
                    if (math.isfinite(requested_v) and
                            math.isfinite(requested_w)):
                        v = clamp(requested_v, -CFG.v_max, CFG.v_max)
                        w = clamp(requested_w, -CFG.w_max, CFG.w_max)
                    else:
                        v = w = 0.0
                    drive_deadline = now + CFG.drive_command_timeout_s
                elif not r_ctrl.readable or now >= drive_deadline:
                    v = w = 0.0

                if r_mode.ready():
                    requested_mode = int(r_mode.data['mode'])
                    if requested_mode not in MODE_NAMES:
                        print("[base] ignoring invalid runtime mode %d" %
                              requested_mode, flush=True)
                    else:
                        requested_lean = float(r_mode.data['lean_angle_deg'])
                        if not math.isfinite(requested_lean):
                            requested_lean = 2.0
                        else:
                            magnitude = clamp(
                                abs(requested_lean), CFG.lean_angle_min_deg,
                                CFG.lean_angle_max_deg)
                            requested_lean = (-magnitude if requested_lean < 0.0
                                              else magnitude)
                        changed = (requested_mode != mode or
                                   (requested_mode == MODE_LEAN and
                                    requested_lean != lean_deg))
                        mode = requested_mode
                        if requested_mode == MODE_LEAN:
                            lean_deg = requested_lean
                        if changed:
                            lean_clear_until = 0.0
                            mode_update = mode
                            mode_update_packets = 5
                        mode_deadline = now + CFG.mode_command_timeout_s

                if r_cmd_int_reset.ready():
                    reset_cmd_integral = bool(
                        r_cmd_int_reset.data['request'])

                if mode_deadline is not None and now >= mode_deadline:
                    fallback_mode = int(CFG.default_mode)
                    fallback_lean = clamp(float(CFG.lean_angle_deg),
                                          CFG.lean_angle_min_deg,
                                          CFG.lean_angle_max_deg)
                    changed = (mode != fallback_mode or
                               (fallback_mode == MODE_LEAN and
                                lean_deg != fallback_lean))
                    mode = fallback_mode
                    if fallback_mode == MODE_LEAN:
                        lean_deg = fallback_lean
                    if changed:
                        lean_clear_until = 0.0
                        mode_update = mode
                        mode_update_packets = 5
                    print("[base] using default mode %s" %
                          MODE_NAMES.get(mode, mode), flush=True)
                    mode_deadline = None

                if mode != prev_mode:
                    print("[base] mode -> %s" % MODE_NAMES.get(mode, mode),
                          flush=True)
                    prev_mode = mode

                # Asserted-but-inactive means the firmware latched lean off.
                lean_request = mode == MODE_LEAN
                if (lean_request
                      and lean_stuck_since is not None
                      and now - lean_stuck_since > LEAN_STUCK_S
                      and now >= lean_next_attempt
                      and abs(last_pitch) < LEAN_SAFE_PITCH_RAD):
                    lean_clear_until = now + LEAN_CLEAR_S
                    lean_next_attempt = now + LEAN_COOLDOWN_S
                    lean_relatch_count += 1
                    mode_update = MODE_BALANCE
                    mode_update_packets = 5
                    print("[base] lean latched off by an abort -- releasing the "
                          "request for %.0f ms to clear it (attempt %d)"
                          % (LEAN_CLEAR_S * 1000.0, lean_relatch_count),
                          flush=True)
                elif (lean_request and lean_clear_until != 0.0 and
                      now >= lean_clear_until):
                    lean_clear_until = 0.0
                    mode_update = MODE_LEAN
                    mode_update_packets = 5

                if max(abs(v), abs(w)) > 1e-4:
                    iq_cmd_active = now

                if now >= next_send:
                    send_twist_tune = (mode == MODE_TWIST and
                                       mode_update_packets == 0)
                    link.send(v, w,
                              runtime_mode=(mode_update
                                            if mode_update_packets > 0
                                            else None),
                              lean_deg=lean_deg,
                              live=True,
                              torque_limit_nm=CFG.action_scale_nm,
                              ff_tau_s_nm=(ff_tau_s_nm if send_twist_tune and
                                           twist_tune_phase == 0 else None),
                              ff_tau_k_nm=(ff_tau_k_nm if send_twist_tune and
                                           twist_tune_phase == 0 else None),
                              ff_yaw_nm=(ff_yaw_nm if send_twist_tune and
                                         twist_tune_phase == 1 else None),
                              profile_amax=(profile_amax if send_twist_tune and
                                            twist_tune_phase == 2 else None),
                              profile_jmax_acc=(profile_jmax_acc
                                                if send_twist_tune and
                                                twist_tune_phase == 2 else None),
                              profile_jmax_dec=(profile_jmax_dec
                                                if send_twist_tune and
                                                twist_tune_phase == 3 else None),
                              profile_enabled=(profile_enabled
                                               if send_twist_tune and
                                               twist_tune_phase == 3 else None),
                              reset_cmd_integral=reset_cmd_integral)
                    reset_cmd_integral = False
                    if send_twist_tune:
                        twist_tune_phase = (twist_tune_phase + 1) % 4
                    if mode_update_packets > 0:
                        mode_update_packets -= 1
                        if mode_update_packets == 0:
                            mode_update = None
                    next_send += period
                    # Skip catch-up bursts after a long host stall.
                    if now - next_send > 0.05:
                        next_send = now + period

                for st in link.read(
                        max(0.0, min(next_send - time.monotonic(), 0.05))):
                    if link.imu_diag is not None and link.imu_diag[1] != last_imu_diag_ms:
                        last_imu_diag_ms = link.imu_diag[1]
                        with w_imu_diag.buf() as b: b['words'] = link.imu_diag
                    status_count += 1
                    last_pitch = st.pitch
                    if mode is None:
                        mode = st.runtime_mode

                    # Lean inactivity indicates a latch only while torque is live.
                    if mode == MODE_LEAN and st.live and not st.lean_active:
                        if lean_stuck_since is None:
                            lean_stuck_since = now
                    else:
                        lean_stuck_since = None

                    # Log torque transitions with their missing prerequisites.
                    if st.live != prev_live:
                        if not st.live:
                            print("[base] DISARM tick=%d pitch=%+.2fdeg "
                                  "missing=%s err=%#x/%#x" %
                                  (st.tick, math.degrees(st.pitch),
                                   ",".join(st.missing) or "(none)",
                                   st.ax0_err, st.ax1_err), flush=True)
                        else:
                            print("[base] ARM    tick=%d pitch=%+.2fdeg" %
                                  (st.tick, math.degrees(st.pitch)), flush=True)
                        prev_live = st.live

                    # Publish sign-corrected [left, right], not raw ODrive axis order.
                    _ax_pos = (st.pos0, st.pos1)
                    _ax_vel = (st.vel0, st.vel1)
                    _ax_iq = (st.iq[0], st.iq[1])
                    _li, _ri = CFG.left_axis, CFG.right_axis
                    # Feedback signs are the inverse of the firmware command signs.
                    _sl = -float(base_config['sign_dir_left'])
                    _sr = -float(base_config['sign_dir_right'])
                    with w_state.buf() as b:
                        b['pos'] = (_ax_pos[_li] * _sl, _ax_pos[_ri] * _sr)
                        b['vel'] = (_ax_vel[_li] * _sl, _ax_vel[_ri] * _sr)
                        b['torque'] = (0.0, st.torque1)
                        b['ff'] = (0.0, 0.0)
                        # ctrl is wheel velocity in twist mode and policy torque otherwise.
                        if mode == MODE_TWIST:
                            _half = 0.5 * base_config['robot_width_m']
                            b['ctrl'] = (v - (w * _half), v + (w * _half))
                        else:
                            _ax_act = (st.act0, st.act1)
                            b['ctrl'] = (_ax_act[_li], _ax_act[_ri])
                        # Current is unsigned and ordered [left, right].
                        if now - iq_cmd_active > IQ_ZERO_AFTER_S:
                            b['iq'] = (0.0, 0.0)
                        else:
                            b['iq'] = (_ax_iq[_li], _ax_iq[_ri])
                        b['gains'] = applied_gains
                        b['pos_estimate'] = 0.0
                        b['yaw_error'] = 0.0

                    # Publish the balancer's filtered orientation in degrees.
                    with w_orient.buf() as b:
                        b['rpy'] = tuple(math.degrees(a) for a in st.rpy)
                    with w_raw.buf() as b:
                        b['accel'] = st.accel
                        b['gyro'] = st.gyro

                    if now >= next_health:
                        loop_hz = status_count / 1.0
                        status_count = 0
                        next_health += 1.0
                        if now >= next_status:
                            with w_status.buf() as b:
                                b['voltage'] = st.vbus
                                b['errors'] = (float(st.ax1_err), float(st.ax0_err))
                                b['loop_hz'] = loop_hz
                            next_status += 10.0
                        with w_health.buf() as b:
                            b['control_tick'] = st.tick
                            b['missed_deadlines'] = st.missed
                            b['exec_cycles'] = st.exec_cycles
                            b['axis0_error'] = st.ax0_err
                            b['axis1_error'] = st.ax1_err
                            b['can_rx_lost'] = st.rx_lost
                            b['can_rx_fifo_full'] = st.fifo_full
                            b['can_hb_stale'] = st.hb_stale
                            b['flags'] = st.flags
                            b['link_crc_errors'] = link.crc_errors
                        print(f"[base] status={loop_hz:.1f}Hz tick={st.tick} "
                              f"missed={st.missed} "
                              f"err={st.ax0_err:#x}/{st.ax1_err:#x} "
                              f"rxlost={st.rx_lost} crc={link.crc_errors} "
                              f"{'LIVE' if st.live else 'shadow'}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        # A normal daemon exit hands control back to zero-command balance.
        link.close()
        print("[base] stopped", flush=True)
