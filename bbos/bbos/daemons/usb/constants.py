"""Config and record type for the usb daemon."""

from bbos import register, realtime


# ============================================================================
# Config
# ============================================================================
@register
class usb:
    """Tunables for the usb daemon, read once at startup."""

    # --- Event log ----------------------------------------------------------
    # No dot, no .log: the bb CLI wipes /dev/shm/usb.* and *.log on restart.
    log_path: str = "/dev/shm/usb_events"
    log_bytes: int = 1 << 20
    log_evict_frac: float = 0.25
    tree_bytes: int = 8192    # Field width; a 16-device tree is 2.2 KB.
    tree_buf_ms: int = 2560   # 256 versions; cascades are ~10 ms apart.

    # --- Physical connectors ------------------------------------------------
    # Two port objects per USB-A jack: USB3 shows on the SuperSpeed half only.
    physical_ports: dict = {
        "usb_a": {
            "usb-a-1": ("1-2-port1", "2-1-port1"),
            "usb-a-2": ("1-2-port2", "2-1-port2"),
            "usb-a-3": ("1-2-port3", "2-1-port3"),
            "usb-a-4": ("1-2-port4", "2-1-port4"),
        },
        "usb_c": {  # A device controller, not a host port.
            "usb-c-1": "3550000.usb",
        },
    }


# ============================================================================
# Types
# ============================================================================
@realtime(ms=10)
def usb_tree():
    """One entry per uevent, so this is a stream, not a state.

    Readers must open it sync=True; the default mode jumps to the newest slot
    and drops the burst. shape_hash covers the WHOLE document, so a devnum
    change is a new version.
    """
    return [("json", f"S{usb.tree_bytes}", ()), ("shape_hash", "S32", ())]
