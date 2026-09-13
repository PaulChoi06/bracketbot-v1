import struct, time, os, sys, posix_ipc, mmap, json, curses

META_SIZE = 4096
MAX_TIMELOGS = 128
TIMELOG_HEADER = 64
TIMELOG_SLOT = 128
TIMELOG_SHM_NAME = "bbos_timelogs"
TIMELOG_SIZE = TIMELOG_HEADER + MAX_TIMELOGS * TIMELOG_SLOT

def read_writer_meta(name):
    try:
        shm = posix_ipc.SharedMemory(name)
        m = mmap.mmap(shm.fd, shm.size, mmap.MAP_SHARED, mmap.PROT_READ)
        pid = int.from_bytes(bytes(m[8:12]), 'little')
        meta_len = int.from_bytes(bytes(m[12:16]), 'little')
        meta = json.loads(bytes(m[16:16+meta_len])) if meta_len > 0 else None
        m.close(); shm.close_fd()
        return pid, meta
    except:
        return None, None

def read_timelogs():
    try:
        shm = posix_ipc.SharedMemory(TIMELOG_SHM_NAME)
    except posix_ipc.ExistentialError:
        return []
    m = mmap.mmap(shm.fd, TIMELOG_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    entries = []
    for i in range(MAX_TIMELOGS):
        o = TIMELOG_HEADER + i * TIMELOG_SLOT
        if m[o] != 1:
            continue
        pid = int.from_bytes(bytes(m[o+1:o+5]), 'little')
        name = bytes(m[o+5:o+69]).rstrip(b'\x00').decode(errors='replace')
        avg_ns, std_ns, max_ns = struct.unpack_from('<qqq', m, o + 72)
        entries.append((name, pid, avg_ns, std_ns, max_ns))
    m.close(); shm.close_fd()
    return entries

def scan_writers():
    writers = {}
    try:
        for f in os.listdir('/dev/shm'):
            if f == TIMELOG_SHM_NAME:
                continue
            pid, meta = read_writer_meta(f)
            if meta is not None:
                writers[f] = (pid, meta)
    except:
        pass
    return writers

def fuzzy_match(query, text):
    qi = 0
    for ch in text:
        if qi < len(query) and ch == query[qi]:
            qi += 1
    return qi == len(query)

def topic_searchable(wname, meta):
    s = wname.lower() + " " + meta.get('owner', '').lower()
    for field in meta.get('dtype', []):
        s += " " + field[0].lower()
        if len(field) >= 2:
            s += " " + str(field[1]).lower()
    return s

def run(stdscr):
    curses.curs_set(0)
    curses.use_default_colors()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_GREEN)
    curses.init_pair(6, curses.COLOR_WHITE, -1)
    curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_WHITE)
    stdscr.keypad(True)
    stdscr.timeout(250)

    query = ""
    cursor = 0
    scroll = 0
    expanded = set()
    filtering = False

    while True:
        ch = stdscr.getch()
        if ch == ord('q') and not filtering:
            break
        elif ch == 27:
            if filtering:
                filtering = False
                query = ""
            else:
                break
        elif ch == ord('/') and not filtering:
            filtering = True
            query = ""
        elif filtering and ch in (curses.KEY_BACKSPACE, 127, 8):
            query = query[:-1]
            if not query:
                filtering = False
        elif filtering and ch in (curses.KEY_ENTER, 10, 13):
            filtering = False
        elif filtering and 32 <= ch < 127:
            query += chr(ch)
        elif ch == curses.KEY_UP:
            cursor = max(0, cursor - 1)
        elif ch == curses.KEY_DOWN:
            cursor += 1
        elif ch == curses.KEY_PPAGE:
            cursor = max(0, cursor - 10)
        elif ch == curses.KEY_NPAGE:
            cursor += 10
        elif ch == curses.KEY_HOME:
            cursor = 0
        elif ch in (curses.KEY_ENTER, 10, 13) and not filtering:
            pass  # handled below after building display

        writers = scan_writers()
        timelogs = read_timelogs()
        readers_by_topic = {}
        for name, pid, avg_ns, std_ns, max_ns in timelogs:
            readers_by_topic.setdefault(name, []).append((pid, avg_ns, std_ns, max_ns))

        rows = []
        q = query.lower()
        for wname, (wpid, meta) in sorted(writers.items()):
            if q and not fuzzy_match(q, topic_searchable(wname, meta)):
                continue
            period = meta.get('period')
            owner = meta.get('owner', '?')
            reader_entries = readers_by_topic.get(wname, [])
            best_avg = min((e[1] for e in reader_entries if e[1] > 0), default=0)
            if best_avg > 0:
                freq_str = f"{1e9/best_avg:.0f}Hz"
            elif period:
                freq_str = f"~{1000/period:.0f}Hz"
            else:
                freq_str = "state"
            n_readers = len(reader_entries)
            dtype_fields = meta.get('dtype', [])
            dtype_summary = ", ".join(f[0] for f in dtype_fields if f[0] != 'timestamp')
            rows.append(("writer", wname, freq_str, wpid, owner, dtype_summary, dtype_fields, reader_entries, n_readers))

        orphans = {k: v for k, v in readers_by_topic.items() if k not in writers}
        for rname, entries in sorted(orphans.items()):
            if q and not fuzzy_match(q, rname.lower()):
                continue
            rows.append(("orphan", rname, "", 0, "", "", [], entries, len(entries)))

        h, w = stdscr.getmaxyx()
        stdscr.erase()

        # build display lines: list of (kind, text, toggle_key)
        display = []
        for row in rows:
            kind = row[0]
            if kind == "writer":
                _, wname, freq_str, wpid, owner, dtype_summary, dtype_fields, reader_entries, n_readers = row
                display.append(("topic", f" {wname:<28} {freq_str:>6}  {wpid:>6}  {n_readers:>3}  {dtype_summary}", wname))
                if wname in expanded:
                    display.append(("detail", f"   owner: {owner}", None))
                    for field in dtype_fields:
                        if field[0] == 'timestamp':
                            continue
                        if len(field) == 3:
                            shape = tuple(field[2]) if isinstance(field[2], list) else (field[2],)
                            display.append(("detail", f"   {field[0]}: {field[1]} {shape}", None))
                        else:
                            display.append(("detail", f"   {field[0]}: {field[1]}", None))
                    for rpid, avg_ns, std_ns, max_ns in reader_entries:
                        if avg_ns > 0:
                            freq = 1e9 / avg_ns
                            avg_ms = avg_ns / 1e6
                            std_ms = std_ns / 1e6
                            max_ms = max_ns / 1e6
                            display.append(("reader", f"   -> pid={rpid}  {freq:.1f}Hz  avg={avg_ms:.1f}ms std={std_ms:.1f}ms max={max_ms:.1f}ms", None))
                        else:
                            display.append(("reader", f"   -> pid={rpid}  (waiting)", None))
            else:
                _, rname = row[0], row[1]
                reader_entries = row[7]
                display.append(("orphan", f" {rname:<28} {'---':>6}  {'---':>6}  {len(reader_entries):>3}  (no writer)", rname))
                if rname in expanded:
                    for rpid, avg_ns, std_ns, max_ns in reader_entries:
                        display.append(("orphan_detail", f"   -> pid={rpid}  (disconnected)", None))

        # clamp cursor
        topic_indices = [i for i, (k, _, tk) in enumerate(display) if tk is not None]
        if topic_indices:
            cursor = max(0, min(cursor, len(topic_indices) - 1))
            cursor_display_idx = topic_indices[cursor]
        else:
            cursor = 0
            cursor_display_idx = -1

        # handle enter toggle
        if ch in (curses.KEY_ENTER, 10, 13) and not filtering:
            if 0 <= cursor < len(topic_indices):
                idx = topic_indices[cursor]
                _, _, toggle_key = display[idx]
                if toggle_key:
                    expanded.symmetric_difference_update({toggle_key})

        # ensure cursor row is visible
        body_h = h - 3
        if body_h > 0 and cursor_display_idx >= 0:
            if cursor_display_idx < scroll:
                scroll = cursor_display_idx
            elif cursor_display_idx >= scroll + body_h:
                scroll = cursor_display_idx - body_h + 1
        max_scroll = max(0, len(display) - body_h)
        scroll = max(0, min(scroll, max_scroll))

        # header
        filter_str = f"  /{query}_" if filtering else (f"  /{query}" if query else "")
        header = f" bbos topics  {len(writers)}W {len(timelogs)}R{filter_str}"
        try:
            stdscr.addnstr(0, 0, header.ljust(w), w, curses.color_pair(4) | curses.A_BOLD)
        except curses.error:
            pass

        # column header
        col_hdr = f" {'TOPIC':<28} {'FREQ':>6}  {'PID':>6}  {'#R':>3}  {'FIELDS'}"
        try:
            stdscr.addnstr(1, 0, col_hdr.ljust(w), w, curses.A_BOLD | curses.A_UNDERLINE)
        except curses.error:
            pass

        # render visible rows
        visible = display[scroll:scroll + body_h]
        for i, (kind, text, toggle_key) in enumerate(visible):
            y = i + 2
            if y >= h - 1:
                break
            display_idx = scroll + i
            is_cursor_row = display_idx == cursor_display_idx
            line = text[:w]

            if toggle_key is not None:
                marker = "v" if toggle_key in expanded else ">"
                line = marker + line[1:]

            if is_cursor_row:
                attr = curses.color_pair(7) | curses.A_BOLD
            elif kind == "topic":
                attr = curses.color_pair(6) | curses.A_BOLD
            elif kind == "orphan":
                attr = curses.color_pair(2) | curses.A_BOLD
            elif kind == "reader":
                attr = curses.color_pair(1)
            elif kind == "orphan_detail":
                attr = curses.color_pair(2)
            else:
                attr = curses.color_pair(3)

            try:
                stdscr.addnstr(y, 0, line.ljust(w), w, attr)
            except curses.error:
                pass

        # footer
        scrollbar = ""
        if len(display) > body_h:
            pct = scroll / max_scroll * 100 if max_scroll > 0 else 0
            scrollbar = f"  [{scroll+1}-{min(scroll+body_h, len(display))}/{len(display)}] {pct:.0f}%"
        footer = f" q:quit  /:filter  arrows:select  enter:expand{scrollbar}"
        try:
            stdscr.addnstr(h - 1, 0, footer.ljust(w), w, curses.color_pair(5))
        except curses.error:
            pass

        stdscr.refresh()

def plain():
    writers = scan_writers()
    timelogs = read_timelogs()
    readers_by_topic = {}
    for name, pid, avg_ns, std_ns, max_ns in timelogs:
        readers_by_topic.setdefault(name, []).append((pid, avg_ns, std_ns, max_ns))

    print(f"{'TOPIC':<28} {'FREQ':>6}  {'PID':>6}  {'#R':>3}  FIELDS")
    print("-" * 72)
    for wname, (wpid, meta) in sorted(writers.items()):
        period = meta.get('period')
        owner = meta.get('owner', '?')
        reader_entries = readers_by_topic.get(wname, [])
        best_avg = min((e[1] for e in reader_entries if e[1] > 0), default=0)
        if best_avg > 0:
            freq_str = f"{1e9/best_avg:.0f}Hz"
        elif period:
            freq_str = f"~{1000/period:.0f}Hz"
        else:
            freq_str = "state"
        dtype_fields = meta.get('dtype', [])
        dtype_summary = ", ".join(f[0] for f in dtype_fields if f[0] != 'timestamp')
        print(f"{wname:<28} {freq_str:>6}  {wpid:>6}  {len(reader_entries):>3}  {dtype_summary}")

    orphans = {k: v for k, v in readers_by_topic.items() if k not in writers}
    for rname, entries in sorted(orphans.items()):
        print(f"{rname:<28} {'---':>6}  {'---':>6}  {len(entries):>3}  (no writer)")

def main():
    if "--plain" in sys.argv:
        plain()
    else:
        curses.wrapper(run)

if __name__ == "__main__":
    main()
