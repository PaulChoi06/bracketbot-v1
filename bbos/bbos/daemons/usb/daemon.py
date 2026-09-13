"""usb daemon: publish usb.tree, one version per kernel uevent.

Every event folds into an in-memory device set and publishes a new tree, in
kernel order. Nothing polls: sysfs is fully walked only on cold start and on
proven loss (a SEQNUM skip or ENOBUFS).

TODO(privilege): /dev/kmsg needs root. Open today by a hand-applied
`sysctl kernel.dmesg_restrict=0`; belongs in the image.
"""

import errno
import hashlib
import json
import os
import re
import select
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from bbos import Config, Type, Writer
from ramlog import log, open_log

CFG = Config("usb")


# ============================================================================
# Enums
# ============================================================================
class Action(StrEnum):
    """Every ACTION the kernel can put on a uevent, from kobject.h.

    Listed whole even though we fold four, so what we ignore is visible.
    """

    ADD = "add"            # A kobject appeared.
    REMOVE = "remove"      # Going away; its directory is already gone.
    BIND = "bind"          # A driver attached to an interface.
    UNBIND = "unbind"      # A driver detached.
    CHANGE = "change"      # Changed in place; USB uses it for over-current.
    MOVE = "move"          # Renamed in sysfs.
    ONLINE = "online"      # Host controller came back.
    OFFLINE = "offline"    # Host controller died.


class Trigger(StrEnum):
    """What produced a tree version: the Actions we fold, plus our own reasons.

    A StrEnum so it needs no .value: it formats, logs and JSON-encodes as the
    bare string. The four derived from Action share its values.
    """

    ADD = Action.ADD               # A device enumerated.
    REMOVE = Action.REMOVE         # A device left the bus.
    BIND = Action.BIND             # A driver attached.
    UNBIND = Action.UNBIND         # A driver detached, still enumerated.
    COLD_START = "cold_start"      # First tree, built by walking sysfs.
    REBUILD = "rebuild"            # Walked sysfs again, events may be lost.
    OVER_CURRENT = "over_current"  # A change carrying OVER_CURRENT_PORT.
    FAULT = "fault"                # Link faults counted off kmsg.


# ============================================================================
# Paths and tunables
# ============================================================================
USB_ROOT = Path("/sys/bus/usb/devices")
UDC_ROOT = Path("/sys/class/udc")

# --- Uevent stream ----------------------------------------------------------
# sysfs has no memory; a remove uevent carries the dying devnum.
NETLINK_KOBJECT_UEVENT = 15     # Not exported by socket.
UEVENT_GROUP_KERNEL = 1         # 2 is udev's re-broadcast.
UEVENT_RCVBUF = 8 << 20
UEVENT_MSG_MAX = 1 << 16        # Cap on one recv.
SEQNUM_PATH = Path("/sys/kernel/uevent_seqnum")

# --- Error stream -----------------------------------------------------------
# Errors go to the printk ring, not netlink. kmsg seq is unrelated to SEQNUM.
KMSG_PATH = "/dev/kmsg"
KMSG_MAX_PRI = 4                # Warning and worse.
KMSG_PRI_MASK = 7               # Level is the low 3 bits.
KMSG_DRAIN_MAX = 64             # Per wakeup; a storm must not starve uevents.
KMSG_RECORD_MAX = 8192          # CONSOLE_EXT_LOG_MAX: one read, one record.
MONO_RESAMPLE_S = 1.0           # How often to re-sample the mono->wall offset.

# Only what the physical link caused; -32 EPIPE and -19 ENODEV are routine.
LINK_FAULTS = {
    71: "EPROTO",       # Stopped answering.
    84: "EILSEQ",       # CRC mismatch.
    62: "ETIME",        # No answer in the turn-around.
    75: "EOVERFLOW",    # Babble.
    108: "ESHUTDOWN",   # Controller or device disabled.
}
ENUM_GAVE_UP = "unable to enumerate"  # The hub's verdict, no errno.
ENUM_KEY = "enum"                     # Lowercase: not an errno.
# Anchored: a bare -N would match every port path.
ERRNO = re.compile(
    r"(?:error|status|code|control|failed with)[:= ]+\(?-(\d+)\)?", re.I)

# --- Published tree ---------------------------------------------------------
HASH_CHARS = 32  # Must match shape_hash S32 in constants.py.


# ============================================================================
# sysfs
# ============================================================================
def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


# A node's sysfs fields: a new attribute is one row and no reader changes.
SYSFS_FIELDS = {
    "id": lambda d: f"{_read(d / 'idVendor')}:{_read(d / 'idProduct')}",
    "product": lambda d: _read(d / "product") or None,
    "speed": lambda d: float(_read(d / "speed") or 0),      # Negotiated Mb/s.
    "devnum": lambda d: int(_read(d / "devnum") or 0),      # Bus address.
    "maxchild": lambda d: int(_read(d / "maxchild") or 0),  # 0 if not a hub.
}

# The rest of a node. With SYSFS_FIELDS this is every key a node may carry.
DERIVED_FIELDS = (
    "path",        # The sysfs name, which is also the position in the tree.
    "drivers",     # {interface: driver}, from bind/unbind events.
    "stale",       # Present only when an add was read while we were behind.
    "ports",       # {port number: port}, present only on a hub.
)


def read_node(name):
    """One device's fields, per SYSFS_FIELDS. One directory, not a walk."""
    d = USB_ROOT / name
    return {field: build(d) for field, build in SYSFS_FIELDS.items()}


def port_device(port_name):
    """Return the device a port object would hold.

    1-2-port3 -> 1-2.3, usb1-port2 -> 1-2.
    """
    prefix, sep, num = port_name.rpartition("-port")
    if not sep:
        return None
    if prefix.startswith("usb"):
        return f"{prefix[3:]}-{num}"      # Root hub: usb1-port2 -> 1-2.
    return f"{prefix}.{num}"              # Any other hub: 1-2-port3 -> 1-2.3.


def device_port(name):
    """Invert port_device: 1-2.4 -> 1-2-port4, 1-2 -> usb1-port2.

    A port name is its own answer, so the give-up line and the errors before
    it land on one node.
    """
    if not name or "-port" in name:
        return name or None
    if "." in name:
        hub, _, num = name.rpartition(".")
        return f"{hub}-port{num}"
    bus, _, num = name.partition("-")  # Root hub child: 1-2 -> usb1-port2.
    return f"usb{bus}-port{num}" if num else None


def sysfs_devices():
    """Every usb_device by sysfs name.

    Interfaces carry a ':' and are not devices.
    """
    try:
        return sorted(p.name for p in USB_ROOT.iterdir() if ":" not in p.name)
    except OSError:
        return []


# ============================================================================
# Uevent stream
# ============================================================================
def uevent_socket():
    """Bind the netlink socket the kernel broadcasts uevents on."""
    s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW,
                      NETLINK_KOBJECT_UEVENT)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UEVENT_RCVBUF)
    s.bind((0, UEVENT_GROUP_KERNEL))
    return s


def parse_uevent(payload):
    """Netlink message -> {KEY: VALUE}.

    The `add@/devices/...` summary line carries no '=' and drops out on its
    own.
    """
    fields = {}
    for field in payload.split(b"\x00"):
        key, sep, value = field.partition(b"=")
        if sep:
            key = key.decode("utf-8", "replace")
            fields[key] = value.decode("utf-8", "replace")
    return fields


def product_id(product):
    """PRODUCT=bda/5883/100 -> 0bda:5883.

    Uevents print %x, dropping the leading zero that sysfs idVendor keeps.
    """
    parts = product.split("/")
    if len(parts) < 2:
        return ""
    try:
        return f"{int(parts[0], 16):04x}:{int(parts[1], 16):04x}"
    except ValueError:
        return ""


def device_name(devpath):
    """Return the trailing name of a DEVPATH.

    /devices/.../usb1/1-2/1-2.3/1-2.3.2.1 -> 1-2.3.2.1
    """
    return devpath.rsplit("/", 1)[-1]


# ============================================================================
# Error stream
# ============================================================================
def kmsg_stream():
    """Open at the end, or the boot backlog replays. None if unreadable."""
    try:
        fd = os.open(KMSG_PATH, os.O_RDONLY | os.O_NONBLOCK)
        os.lseek(fd, 0, os.SEEK_END)
    except OSError as e:
        log("kmsg", "unavailable", f"{e}; USB errors will not be logged")
        return None
    return fd


def parse_kmsg(record):
    """`pri,seq,usec,flags;message` plus ` KEY=value` continuation lines."""
    head, sep, rest = record.partition(";")
    fields = head.split(",")
    if not sep or len(fields) < 3:
        return None
    try:
        out = {"pri": int(fields[0]), "seq": int(fields[1]),
               "usec": int(fields[2])}
    except ValueError:
        return None
    msg, *continuation = rest.split("\n")
    out["msg"] = msg.strip()
    for line in continuation:
        key, sep, value = line.strip().partition("=")
        if sep:
            out[key] = value
    return out


def kmsg_device(rec):
    """Return the device a record names, as a tree path.

    Only devices are nodes, so an interface (`1-2.3.3:1.1`) drops its suffix.
    """
    dev = rec.get("DEVICE", "")
    if dev.startswith("+usb:"):
        return dev[len("+usb:"):].split(":", 1)[0]
    head = rec["msg"].split(":", 1)[0].split()
    return head[-1] if len(head) > 1 else ""


# kmsg stamps are monotonic, log lines wall clock. Resampled for NTP steps.
_mono_offset = [0.0, float("-inf")]  # [offset, sampled at].


def mono_to_wall(usec):
    """Convert a kmsg monotonic stamp to a local wall-clock datetime."""
    now = time.monotonic()
    if now - _mono_offset[1] > MONO_RESAMPLE_S:
        _mono_offset[0] = time.time() - now
        _mono_offset[1] = now
    return datetime.fromtimestamp(_mono_offset[0] + usec / 1e6).astimezone()


# ============================================================================
# The tree
# ============================================================================
class UsbEvents(threading.Thread):
    """A device tree folded from the uevent stream, never polled.

    One event in, one version out, in kernel order.

    Single-threaded: cold_start() runs before the thread starts, everything
    after only from run(), so nothing locks.
    """

    def __init__(self, sock, publish, kmsg=None):
        """Start empty; cold_start() seeds the tree before run() folds it."""
        super().__init__(daemon=True, name="uevent")
        self.sock = sock
        self.kmsg = kmsg                  # fd, or None if we cannot read it.
        self.publish = publish            # (version, kind, detail, tree).
        self.devices = {}                 # sysfs name -> node.
        self.over_current = {}            # Port name -> count.
        self.drivers = {}                 # Device name -> {interface: driver}.
        self.udc = {}                     # udc name -> raw sysfs attrs.
        self.last_seq = None
        self.version = 0
        self.applied = 0                  # Events folded into the tree.
        self.gaps = 0
        # Port name -> {code: count}, since we started.
        self.errors = {}
        self.kmsg_seq = None

    # --- The tree ----------------------------------------------------------
    def tree(self):
        """Return the physical-connector view.

        No sysfs: every field came from an event or the last rebuild.
        """
        out = {}
        for kind, ports in CFG.physical_ports.items():
            for label, target in ports.items():
                if kind == "usb_c":
                    out[label] = {"udc": target, **self.udc.get(target, {})}
                else:
                    out[label] = {half: self._port_node(half)
                                  for half in target}
        return out

    def _port_node(self, port_name):
        """One port and, if a hub sits on it, its ports too.

        Recurses on the name alone: a hub's child is its name plus the port
        number.
        """
        name = port_device(port_name)
        dev = self.devices.get(name) if name else None
        node = None
        if dev is not None:
            node = dict(dev, path=name, drivers=self.drivers.get(name, {}))
            kids = {i: self._port_node(f"{name}-port{i}")
                    for i in range(1, (dev.get("maxchild") or 0) + 1)}
            if kids:
                node["ports"] = kids
        return {"over_current": self.over_current.get(port_name, 0),
                "errors": self.errors.get(port_name, {}),
                "device": node}

    def emit(self, kind, detail=""):
        """One event, one version, in the order the kernel produced them."""
        self.version += 1
        self.publish(self.version, kind, detail, self.tree())

    # --- Rebuilding from sysfs ---------------------------------------------
    def rebuild(self, kind, detail=""):
        """Throw the tree away and walk sysfs.

        For when events cannot be trusted: no history at startup, a hole
        after a loss. Leaves last_seq alone, or the queued events behind it
        would each look like a gap.
        """
        devices = {n: read_node(n) for n in sysfs_devices()}
        self.devices = devices
        self.over_current = {p.name: int(_read(p / "over_current_count") or 0)
                             for p in USB_ROOT.glob("*/*-port*")}
        self.drivers = {}
        for iface in USB_ROOT.iterdir():  # Interfaces are the ':' names.
            dev, sep, num = iface.name.partition(":")
            link = iface / "driver"
            if sep and link.is_symlink():
                self.drivers.setdefault(dev, {})[num] = os.path.basename(
                    os.path.realpath(link))
        self.udc = {t: {a: _read(UDC_ROOT / t / a) or None
                        for a in ("state", "current_speed", "maximum_speed")}
                    for t in CFG.physical_ports.get("usb_c", {}).values()}
        self.emit(kind, detail)
        return devices

    def cold_start(self):
        """Read the sequence counter before the walk.

        Whatever happens during it is queued and replays after, and add/remove
        are idempotent against what it found.
        """
        seq = int(_read(SEQNUM_PATH) or 0)
        devices = self.rebuild(Trigger.COLD_START)
        self.last_seq = seq
        return devices

    # --- The event loop -----------------------------------------------------
    def run(self):
        """Drain both streams forever.

        Uevents say what changed, kmsg what went wrong. One thread, so
        nothing here locks.
        """
        readers = {self.sock.fileno(): self.read_uevent}
        if self.kmsg is not None:
            readers[self.kmsg] = self.read_kmsg
        poller = select.poll()
        for fd in readers:
            poller.register(fd, select.POLLIN)
        while True:
            # ENOBUFS raises POLLERR without POLLIN; gating on it would spin.
            for fd, _ in poller.poll():
                readers[fd]()

    def read_uevent(self):
        """Fold one waiting uevent, or rebuild if the kernel dropped some."""
        try:
            payload = self.sock.recv(UEVENT_MSG_MAX)
        except OSError as e:
            # ENOBUFS: the kernel dropped messages, only a walk can recover.
            self.gaps += 1
            log("uevent", "overrun", str(e))
            self.rebuild(Trigger.REBUILD, f"overrun {e}")
            return
        try:
            self.handle(parse_uevent(payload))
        except Exception as e:
            # One malformed event must not kill the thread and freeze the tree.
            log("uevent", "handler", f"{type(e).__name__}: {e}")

    def read_kmsg(self):
        """Drain the records waiting on the kmsg fd, bounded per wakeup.

        A printk storm cannot starve uevents. Level-triggered, so the
        remainder wakes poll() again.
        """
        hit = {}  # Ports faulted in this drain.
        for _ in range(KMSG_DRAIN_MAX):
            try:
                raw = os.read(self.kmsg, KMSG_RECORD_MAX)
            except BlockingIOError:
                break  # Caught up.
            except OSError as e:
                if e.errno != errno.EPIPE:      # EPIPE: the ring overwrote us.
                    log("kmsg", "read", str(e))
                    break
                log("kmsg", "overrun", "ring overwrote unread records")
                continue
            port = self.handle_kmsg(raw)
            if port:
                hit[port] = hit.get(port, 0) + 1
        if hit:  # One version per drain, not per record.
            faults = sorted(hit.items())
            self.emit(Trigger.FAULT,
                      " ".join(f"{p}={n}" for p, n in faults))

    def handle_kmsg(self, raw):
        """Log one record if it is a USB error.

        Returns the port a link fault landed on; the caller makes one tree
        version per drain.
        """
        try:
            rec = parse_kmsg(raw.decode("utf-8", "replace"))
        except Exception as e:  # A bad record must not take the tree down.
            log("kmsg", "parse", f"{type(e).__name__}: {e}")
            return None
        if rec is None:
            return None
        when = mono_to_wall(rec["usec"])
        # Sequenced before filtering: kmsg counts every message, not just ours.
        if self.kmsg_seq is not None and rec["seq"] > self.kmsg_seq + 1:
            log("kmsg", "seqgap", f"{self.kmsg_seq} -> {rec['seq']}",
                when=when)
        self.kmsg_seq = rec["seq"]

        if (rec["pri"] & KMSG_PRI_MASK) > KMSG_MAX_PRI:
            return None
        if (rec.get("SUBSYSTEM") != "usb"
                and not rec["msg"].startswith(("usb ", "uvcvideo "))):
            return None
        name = kmsg_device(rec)
        log(name or "kmsg", "usberr",
            f"pri={rec['pri'] & KMSG_PRI_MASK} kseq={rec['seq']} {rec['msg']}",
            when=when)

        code = ERRNO.search(rec["msg"])
        if code:
            key = LINK_FAULTS.get(int(code.group(1)))
            if key is None:  # Not a fault we count.
                return None
        elif ENUM_GAVE_UP in rec["msg"]:
            key = ENUM_KEY
        else:
            return None
        port = device_port(name)
        if port is None:  # A fault we cannot pin on a port.
            return None
        counts = self.errors.setdefault(port, {})
        counts[key] = counts.get(key, 0) + 1
        return port

    def handle(self, ev):
        """Fold one uevent into the tree.

        Publishes a version if it changed anything. Everything else still
        counts toward the sequence.
        """
        seq = int(ev.get("SEQNUM", 0) or 0)

        # --- Sequence ---
        # Counter is global; only counting all subsystems makes a skip real.
        expected, self.last_seq = self.last_seq, seq
        if expected is not None and seq > expected + 1:
            skipped = seq - expected - 1
            self.gaps += 1
            log("uevent", "seqgap", f"{expected} -> {seq} (skipped {skipped})")
            self.rebuild(Trigger.REBUILD, f"seqgap skipped={skipped}")

        if ev.get("SUBSYSTEM") != "usb":
            return

        # --- Over-current ---
        # Reported on the hub's interface, so it precedes the device filter.
        if "OVER_CURRENT_PORT" in ev:
            port = os.path.basename(ev["OVER_CURRENT_PORT"])
            count = int(ev.get("OVER_CURRENT_COUNT", 0) or 0)
            self.applied += 1
            self.over_current[port] = count
            log(port, "overcurrent", f"seq={seq} count={count}")
            self.emit(Trigger.OVER_CURRENT, f"{port} count={count}")
            return

        action = ev.get("ACTION")

        # --- Driver bind/unbind ---
        # A device can enumerate fine and still have no driver attached.
        if (ev.get("DEVTYPE") == "usb_interface"
                and action in (Action.BIND, Action.UNBIND)):
            dev, _, num = device_name(ev.get("DEVPATH", "")).partition(":")
            self.applied += 1
            if action == Action.BIND:
                driver = ev.get("DRIVER", "?")
                self.drivers.setdefault(dev, {})[num] = driver
            else:
                # An unbind carries no DRIVER, but we recorded it on the bind.
                driver = self.drivers.get(dev, {}).pop(num, "?")
            log(dev, action, f"seq={seq} interface={num} driver={driver}")
            self.emit(
                Trigger.BIND if action == Action.BIND else Trigger.UNBIND,
                f"{dev}:{num} {driver}")
            return

        # --- Host controller online/offline ---
        # The root hub is not in the tree, so there is nothing to fold.
        if action in (Action.OFFLINE, Action.ONLINE):
            log(device_name(ev.get("DEVPATH", "")), action,
                f"seq={seq} host controller")
            return

        if ev.get("DEVTYPE") != "usb_device":
            return  # Interfaces carry no devnum.
        name = device_name(ev.get("DEVPATH", ""))
        devnum = int(ev.get("DEVNUM", 0) or 0)
        dev_id = product_id(ev.get("PRODUCT", ""))

        # --- add ---
        # The only sysfs read here; a moved-on devnum means we are behind.
        if action == Action.ADD:
            self.applied += 1
            node = read_node(name)
            live = node["devnum"]           # What sysfs says right now.
            node["devnum"] = devnum         # What the event said.
            if live != devnum:
                node["speed"], node["maxchild"], node["stale"] = 0.0, 0, True
                log(name, "behind",
                    f"seq={seq} event devnum={devnum} but sysfs says {live}")
            self.devices[name] = node
            log(name, Trigger.ADD,
                f"seq={seq} id={dev_id} devnum={devnum} "
                f"speed={node['speed']:.0f} maxchild={node['maxchild']}")
            self.emit(Trigger.ADD, f"{name} devnum={devnum}")

        # --- remove ---
        # No sysfs read: the directory is gone; this is devnum's only record.
        elif action == Action.REMOVE:
            self.applied += 1
            self.devices.pop(name, None)
            self.drivers.pop(name, None)
            log(name, Trigger.REMOVE, f"seq={seq} id={dev_id} devnum={devnum}")
            self.emit(Trigger.REMOVE, f"{name} devnum={devnum}")


# ============================================================================
# Daemon
# ============================================================================
def main():
    """Open the topic and the log, seed the tree, then run until killed."""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    # Unwind, so shm unlinks.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # keeptime=False: the period only sizes the ring; pacing would throttle us.
    with Writer("usb.tree", Type("usb_tree"),
                keeptime=False, buf_ms=CFG.tree_buf_ms) as w_tree:
        # After the Writer, so a losing duplicate never touches the live log.
        open_log(CFG.log_path, CFG.log_bytes, CFG.log_evict_frac)

        def publish(version, kind, detail, tree):
            """Log every version; the topic ring carries them to readers."""
            blob = json.dumps(tree, separators=(",", ":"))
            log("tree", f"v{version}", f"{kind} {detail} | {blob}")
            if len(blob) > CFG.tree_bytes:
                log("tree", "overflow",
                    f"{len(blob)} > {CFG.tree_bytes} bytes")
                return
            digest = hashlib.sha256(blob.encode()).hexdigest()[:HASH_CHARS]
            with w_tree.buf() as b:
                b["json"] = blob.encode()
                b["shape_hash"] = digest.encode()

        # Bind before the first walk so nothing that happens during it is lost.
        events = UsbEvents(uevent_socket(), publish, kmsg_stream())
        seeded = events.cold_start()
        log("uevent", "start", f"{len(seeded)} devices, seq={events.last_seq}")
        events.start()

        # Inside: Writer.__exit__ swallows and tracebacks.
        try:
            events.join()  # Nothing to do here; the thread does the work.
        except (KeyboardInterrupt, SystemExit):
            print("[+] shutting down")


if __name__ == "__main__":
    main()
