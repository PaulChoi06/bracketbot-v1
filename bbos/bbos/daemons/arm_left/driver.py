import scservo_sdk as scs
import numpy as np
import os
from bbos import Config
CFG = Config(os.path.basename(os.getcwd()))

# http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SCSCL-emanual-cbcc8ab2e3384282a01d4bf3
FIRMWARE_MAJOR_VERSION = (0, 1)
FIRMWARE_MINOR_VERSION = (1, 1)
MODEL_NUMBER = (3, 2)
PROTOCOL = 0
SERIAL_READ_TIMEOUT_S = 0.004  # SDK opens the port with timeout=0, which busy-spins in rxPacket
HLS_VELOCITY_RAW_TO_TURNS_PER_SEC = 0.732 / 60.0

# Phase register (address 18) bit map (Feetech STS/HLS datasheet section 3.1):
#   BIT7 Servo direction   (0 positive / 1 CCW)
#   BIT6 PWM mode          (0 high-freq / 1 low-freq)  -- clear for 24kHz (quiet)
#   BIT5 Voltage sampling  (0 1.5K low-V / 1 1K high-V)
#   BIT4 Feedback mode     (0 single-turn / 1 full-range multi-turn)
#   BIT3 Speed mode        (0 speed_0=stop / 1 speed_0=max)
#   BIT2 Speed unit        (0 50 steps/s / 1 steps/s)
#   BIT1 Drive bridge      (0 brushless / 1 brushed)
#   BIT0 Phase coefficient (0 positive / 1 negative) -- factory-calibrated commutation polarity
PHASE_BIT4_MULTITURN = 0x10
PHASE_BIT6_PWM_16KHZ = 0x40
PHASE_BIT7_DIRECTION = 0x80

# HLS servo control table (Model 4618)
# Reference: FT-SCS custom protocol for HLS series
SCS_SERIES_CONTROL_TABLE = {
    # EPROM (0x00-0x27) - persistent, copied to SRAM on power-up
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,  # read-only
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,  # read-only
    "Model_Number": MODEL_NUMBER,  # read-only
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Deputy_ID": (7, 1),  # HLS: secondary ID, not return delay time!
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Also101_Condition": (20, 1),
    "P_Coefficient": (21, 1),  # EPROM P - copied to Kp(50) on power-up
    "D_Coefficient": (22, 1),  # EPROM D - copied to Kd(51) on power-up
    "I_Coefficient": (23, 1),  # EPROM I - copied to Ki(52) on power-up
    "Minimum_Startup_Torque": (24, 1),  # HLS: 1 byte, not 2!
    "Integral_Limit": (25, 1),  # HLS: max integral value = limit*4
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angle_Resolution": (30, 1),
    "Position_Offset": (31, 2),  # HLS: position offset, BIT15 = sign
    "Operating_Mode": (33, 1),  # 0=position, 1=velocity, 2=current, 3=PWM
    "Current_Loop_P": (34, 1),  # HLS: current loop P coefficient
    "Current_Loop_I": (35, 1),  # HLS: current loop I coefficient
    "Speed_Loop_P": (37, 1),
    "Overcurrent_Protection_Time": (38, 1),
    "Speed_Loop_I": (39, 1),
    # SRAM (0x28-0x37) - runtime control
    "Torque_Enable": (40, 1),  # 0=off, 1=on, 2=stop
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Target_Current": (44, 2),  # HLS: max running torque (not Running_Time!)
    "Running_Time": (44, 2),  # ST-3120: running time in ms; 0 means use Goal_Velocity
    "Goal_Velocity": (46, 2),  # HLS: 0 = stop, must be non-zero for motion!
    "Torque_Limit": (48, 2),
    "Kp": (50, 1),  # SRAM P coefficient - runtime PID
    "Kd": (51, 1),  # SRAM D coefficient - runtime PID
    "Ki": (52, 1),  # SRAM I coefficient - runtime PID
    "Lock": (55, 1),
    # SRAM feedback (0x38-0x48) - read-only
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),  # PWM duty cycle, BIT10 = direction
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Sync_Write_Flag": (64, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Goal_Position_Readback": (67, 2),  # current target position
    "Present_Current": (69, 2),
    # Factory (0x4D-0x56) - read-only
    "vFk": (77, 1),
    "vKgI": (78, 1),
    "pFk": (79, 1),
    "Moving_Velocity_Threshold": (80, 1),
    "DTs": (81, 1),
    "eFk": (82, 1),
    "Vk": (83, 1),
    "Max_Velocity_Limit": (84, 1),
    "Acceleration_Limit": (85, 1),
    "Acceleration_Multiplier": (86, 1),
    # Aliases for backward compatibility
    "Return_Delay_Time": (7, 1),  # alias, but HLS uses this as Deputy_ID
}

# Sign-magnitude conversion (matches SDK scs_toscs/scs_tohost with b=15 for 16-bit)
def scs_toscs16(x):
    """Convert signed int to sign-magnitude format (bit 15 = sign)"""
    x = np.asarray(x, dtype=np.int32)
    return np.where(x < 0, (-x).astype(np.uint16) | 0x8000, x).astype(np.uint16)

def scs_tohost16(x):
    """Convert sign-magnitude format to signed int (bit 15 = sign)"""
    x = np.asarray(x, dtype=np.uint16)
    sign_bit = x & 0x8000
    magnitude = x & 0x7FFF
    return np.where(sign_bit, -magnitude.astype(np.int32), magnitude).astype(np.float32)

def clamp_j0(x):
    """Clamp j0 to ±7 turns before writing"""
    x = np.asarray(x, dtype=np.float32).copy()
    x[0] = np.clip(x[0], -7, 7)
    return x

WRITE_FUNC_TABLE = {
    # HLS: Goal_Position in 0.087 degree units (4096 steps/turn), BIT15 = sign
    "Goal_Position": lambda x: scs_toscs16(np.round(clamp_j0(x) * 4096).astype(np.int32)),
    # HLS: Goal_Velocity is raw value in 0.732 RPM units, NOT scaled by 4096!
    # Don't apply any transformation - pass value directly
    "Target_Current": lambda x: scs_toscs16(np.clip(np.round(np.asarray(x, dtype=np.float32) / 0.0065), -32767, 32767).astype(np.int32)),
    # conversion is raw*0.0065A
}

READ_FUNC_TABLE = {
    # Multi-turn: With BIT4 of Phase set, Present_Position returns full multi-turn range
    # Convert signed steps to turns
    "Present_Position": lambda x: scs_tohost16(x) / 4096.0,
    "Present_Velocity": lambda x: scs_tohost16(x) * HLS_VELOCITY_RAW_TO_TURNS_PER_SEC,
    "Present_Current": lambda x: scs_tohost16(x) * 0.0065,
}

def init_feetech():
    """Initialize feetech connection and configure motors"""
    port = scs.PortHandler(CFG.port)
    packet = scs.PacketHandler(PROTOCOL)
    
    print(f"Connecting to {CFG.port} at {CFG.baudrate} baud")
    
    if not port.openPort():
        raise RuntimeError(f"Failed to open {CFG.port}")
    if not port.setBaudRate(CFG.baudrate):
        raise RuntimeError("Failed to set baudrate")
    port.ser.timeout = SERIAL_READ_TIMEOUT_S
    print(f"Connected to {CFG.port} at {CFG.baudrate} baud")
    # Verify all motors are connected
    connected_motors = []
    for sid in CFG.motors:
        model, comm, err = packet.ping(port, sid)
        if comm == scs.COMM_SUCCESS:
            print(f"ID {sid}: Model={model}, Error={err}")
            connected_motors.append(sid)
        else:
            print(f"ID {sid}: No response ({packet.getTxRxResult(comm)})")
    
    if len(connected_motors) != len(CFG.motors):
        missing = set(CFG.motors) - set(connected_motors)
        raise RuntimeError(f"Motors not found: {missing}")
    return port, packet

def write_motors(port, packet, register_name, values, mask=None):
    """Write values to a specific register on selected motors using group sync write
    
    Args:
        mask: optional boolean array - only write to motors where mask is True
              If None, writes to all motors.
    """
    if len(values) != len(CFG.motors):
        raise ValueError(f"Values array length ({len(values)}) must match motor_ids length ({len(CFG.motors)})")
    
    # Apply write function if available
    if register_name in WRITE_FUNC_TABLE:
        values = WRITE_FUNC_TABLE[register_name](values)
    
    # Get register address and data length from control table
    register_addr = SCS_SERIES_CONTROL_TABLE[register_name][0]
    data_length = SCS_SERIES_CONTROL_TABLE[register_name][1]
    
    group_sync_write = scs.GroupSyncWrite(port, packet, register_addr, data_length)
    
    # Add data for each motor (only if mask allows)
    for i, sid in enumerate(CFG.motors):
        if mask is not None and not mask[i]:
            continue
        value = int(values[i])
        
        if data_length == 1:
            # 1-byte register
            data = [value & 0xFF]
        elif data_length == 2:
            # 2-byte register (little-endian)
            data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
        else:
            raise ValueError(f"Unsupported data length: {data_length}")
        
        group_sync_write.addParam(sid, data)
    
    # Execute group write (only if we have motors to write to)
    if mask is None or np.any(mask):
        group_sync_write.txPacket()
    group_sync_write.clearParam()
    return True


def read_motors(port, packet, register_name, result_array):
    """Read a specific register from all motors using group sync read"""
    
    # Get register address and data length from control table
    register_addr = SCS_SERIES_CONTROL_TABLE[register_name][0]
    data_length = SCS_SERIES_CONTROL_TABLE[register_name][1]
    func = READ_FUNC_TABLE[register_name] if register_name in READ_FUNC_TABLE else None
    
    group_sync_read = scs.GroupSyncRead(port, packet, register_addr, data_length)
    
    # Add all motors to group read
    for sid in CFG.motors:
        group_sync_read.addParam(sid)
    
    # Execute group read
    comm = group_sync_read.txRxPacket()
    if comm != scs.COMM_SUCCESS:
        print(f"Group sync read failed: {packet.getTxRxResult(comm)}")
        return
    
    # Extract data for each motor
    for i, sid in enumerate(CFG.motors):
        if group_sync_read.isAvailable(sid, register_addr, data_length):
            data = group_sync_read.getData(sid, register_addr, data_length)
            result_array[i] = float(data)
        else:
            result_array[i] = np.nan
    
    # Apply vectorized function to all results if provided
    if func:
        result_array[:] = func(result_array)
    
    group_sync_read.clearParam()


def read_motors_block(port, packet, register_names, result_arrays):
    """Read several registers from all motors in a single group sync read spanning them"""
    regs = [SCS_SERIES_CONTROL_TABLE[r] for r in register_names]
    start = min(a for a, _ in regs)
    length = max(a + l for a, l in regs) - start

    group_sync_read = scs.GroupSyncRead(port, packet, start, length)
    for sid in CFG.motors:
        group_sync_read.addParam(sid)

    comm = group_sync_read.txRxPacket()
    if comm != scs.COMM_SUCCESS:
        print(f"Group sync read failed: {packet.getTxRxResult(comm)}")
        return

    for register_name, (register_addr, data_length), result_array in zip(register_names, regs, result_arrays):
        for i, sid in enumerate(CFG.motors):
            # a short/desynced frame passes the SDK's isAvailable (it checks only the register
            # width, not its offset in the span) and getData would IndexError on it
            if len(group_sync_read.data_dict.get(sid, ())) == length and \
                    group_sync_read.isAvailable(sid, register_addr, data_length):
                result_array[i] = float(group_sync_read.getData(sid, register_addr, data_length))
            else:
                result_array[i] = np.nan
        func = READ_FUNC_TABLE.get(register_name)
        if func:
            result_array[:] = func(result_array)

    group_sync_read.clearParam()


def set_operating_mode(port, packet, values, mask=None, max_retries=10):
    """Write Operating_Mode with retry until verified. Saves/restores torque state."""
    import time
    values = np.asarray(values, dtype=np.uint8)
    prev_torque = np.zeros(len(CFG.motors), dtype=np.float32)
    read_motors(port, packet, "Torque_Enable", prev_torque)
    active = np.ones(len(CFG.motors), dtype=np.bool_) if mask is None else np.asarray(mask, dtype=np.bool_)
    target_ids = [CFG.motors[i] for i in range(len(CFG.motors)) if active[i]]
    target_vals = [int(values[i]) for i in range(len(CFG.motors)) if active[i]]
    print(f"set_operating_mode: motors {target_ids} -> {target_vals}, prev_torque={prev_torque[active].astype(int).tolist()}", flush=True)
    for attempt in range(max_retries):
        write_motors(port, packet, "Torque_Enable", np.zeros(len(CFG.motors), dtype=np.uint8), mask=active)
        time.sleep(0.1)
        write_motors(port, packet, "Lock", np.zeros(len(CFG.motors), dtype=np.uint8), mask=active)
        time.sleep(0.1)
        write_motors(port, packet, "Operating_Mode", values, mask=active)
        time.sleep(0.1)
        write_motors(port, packet, "Lock", np.ones(len(CFG.motors), dtype=np.uint8), mask=active)
        time.sleep(0.1)
        readback = np.zeros(len(CFG.motors), dtype=np.float32)
        read_motors(port, packet, "Operating_Mode", readback)
        failed = active & (readback.astype(np.uint8) != values)
        if not np.any(failed):
            print(f"set_operating_mode: OK on attempt {attempt+1}", flush=True)
            break
        ids = [CFG.motors[i] for i in range(len(CFG.motors)) if failed[i]]
        got = readback[failed].astype(int).tolist()
        want = values[failed].tolist()
        print(f"set_operating_mode: attempt {attempt+1}/{max_retries} FAILED motors {ids} got={got} want={want}", flush=True)
        active = failed
    else:
        ids = [CFG.motors[i] for i in range(len(CFG.motors)) if active[i]]
        raise RuntimeError(f"set_operating_mode: FAILED after {max_retries} retries for motors {ids}")
    restore = prev_torque.astype(np.uint8)
    write_motors(port, packet, "Torque_Enable", restore, mask=mask)
    print(f"set_operating_mode: torque restored to {restore[active if mask is None else np.asarray(mask, dtype=np.bool_)].tolist()}", flush=True)
    time.sleep(0.1)


def configure_motor_phase(port, packet):
    """Configure Phase register (addr 18) per datasheet bit procedure: clear BIT7 (→positive
    servo direction), clear BIT6 (→24kHz), set BIT4 (→multi-turn feedback). Read-modify-write,
    guarded and idempotent.
    """
    import time
    addr = SCS_SERIES_CONTROL_TABLE["Phase"][0]
    for sid in CFG.motors:
        phase, comm, _ = packet.read1ByteTxRx(port, sid, addr)
        if comm != scs.COMM_SUCCESS:
            print(f"Motor {sid}: Phase read failed: {packet.getTxRxResult(comm)}")
            continue

        new = phase
        if new & PHASE_BIT7_DIRECTION:      # BIT7 set (CCW) → subtract 128 → positive direction
            new -= PHASE_BIT7_DIRECTION
        if new & PHASE_BIT6_PWM_16KHZ:      # BIT6 set (low-freq) → subtract 64 → high-freq (24kHz)
            new -= PHASE_BIT6_PWM_16KHZ
        if not (new & PHASE_BIT4_MULTITURN):  # BIT4 clear (single-turn) → add 16 → multi-turn
            new += PHASE_BIT4_MULTITURN

        if new == phase:
            print(f"Motor {sid}: Phase already correct (0x{phase:02X})")
            continue

        comm, _ = packet.write1ByteTxRx(port, sid, addr, new)
        time.sleep(0.03)  # EPROM flash commit
        if comm != scs.COMM_SUCCESS:
            print(f"Motor {sid}: Phase write failed: {packet.getTxRxResult(comm)}")
            continue
        readback, comm2, _ = packet.read1ByteTxRx(port, sid, addr)
        if comm2 == scs.COMM_SUCCESS and readback == new:
            print(f"Motor {sid}: Phase 0x{phase:02X} -> 0x{new:02X}")
        else:
            print(f"Motor {sid}: Phase MISMATCH! wrote=0x{new:02X} readback=0x{readback:02X}")
