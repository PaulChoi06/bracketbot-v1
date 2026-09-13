{ daemonsPath ? "/home/bracketbot/bbos/bbos/daemons"
, bbosRoot ? "/home/bracketbot/bbos"
}:
let
pkgs = import (fetchTarball {
  url = "https://github.com/NixOS/nixpkgs/archive/63dacb46bf939521bdc93981b4cbb7ecb58427a0.tar.gz";
  sha256 = "sha256:1lr1h35prqkd1mkmzriwlpvxcb34kmhc9dnr48gkm8hh089hifmx";
}) {};
in
let
  makeWrapper = pkgs.makeWrapper;
  python = pkgs.python311.withPackages (ps: [ ps.posix_ipc ps.numpy]);
  runtimePath = pkgs.lib.makeBinPath [ python pkgs.busybox ];
  src = builtins.filterSource (path: _: baseNameOf path != "result") ./.;

  manager = pkgs.stdenv.mkDerivation {
    name = "manager";
    version = "1.0";
    inherit src;
    nativeBuildInputs = [ makeWrapper ];
    installPhase = ''
      mkdir -p $out/bin
      ln -s "$src/manager.py" "$out/bin/manager.py"

      makeWrapper ${python}/bin/python3 $out/bin/manager \
        --add-flags "$out/bin/manager.py" \
        --set PATH ${runtimePath}
    '';
  };

  login = pkgs.writeShellApplication {
    name = "login";
    runtimeInputs = [ pkgs.curl pkgs.jq pkgs.coreutils ];
    text = ''
      API_BASE="''${BB_API_URL:-https://api.bracketbot.com}"
      HOSTNAME_VAL="$(cat /proc/sys/kernel/hostname)"
      SERIAL="$(tr -d '\0' < /proc/device-tree/serial-number 2>/dev/null || true)"
      if [ -z "$SERIAL" ]; then
        echo "Error: could not read serial from /proc/device-tree/serial-number"
        exit 1
      fi

      BANNER_LOGIN="─── bb-cloud login ──────────────────────────────────"
      BANNER_LINKED="─── linked! ─────────────────────────────────────────"
      BANNER_EXPIRED="─── link expired ────────────────────────────────────"
      BANNER_TIMEOUT="─── timed out ───────────────────────────────────────"
      SPIN=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)

      echo "$BANNER_LOGIN"
      echo

      REQ_BODY=$(jq -nc --arg s "$SERIAL" --arg h "$HOSTNAME_VAL" '{serial:$s, hostname:$h}')
      RESP=$(curl -fsS -X POST "$API_BASE/v1/robots/link-request" \
        -H "Content-Type: application/json" \
        -d "$REQ_BODY") || { echo "Error: failed to reach $API_BASE"; exit 1; }

      CODE=$(echo "$RESP" | jq -r '.code // empty')
      URL=$(echo "$RESP"  | jq -r '.url  // empty')
      if [ -z "$CODE" ] || [ -z "$URL" ]; then
        echo "Error: unexpected response from server: $RESP"
        exit 1
      fi

      echo "  open in your browser:"
      echo "  $URL"
      echo

      DEADLINE=$(( $(date +%s) + 600 ))
      tick=0
      spin_i=0
      while [ "$(date +%s)" -lt "$DEADLINE" ]; do
        printf '\r  %s waiting for confirmation' "''${SPIN[spin_i]}"
        spin_i=$(( (spin_i + 1) % ''${#SPIN[@]} ))
        sleep 0.1
        tick=$(( tick + 1 ))
        # Poll the API every ~2s (every 20 spinner ticks)
        if [ $(( tick % 20 )) -ne 0 ]; then
          continue
        fi

        STATUS_RESP=$(curl -sS -w '\n%{http_code}' "$API_BASE/v1/robots/link-status?code=$CODE") || continue
        BODY=$(echo "$STATUS_RESP" | sed '$d')
        HTTP=$(echo "$STATUS_RESP" | tail -n1)

        if [ "$HTTP" = "410" ]; then
          printf '\r\033[K'
          echo "$BANNER_EXPIRED"
          echo "  Run 'login' again."
          exit 1
        fi
        if [ "$HTTP" != "200" ]; then
          continue
        fi

        STATUS=$(echo "$BODY" | jq -r '.status // empty')
        if [ "$STATUS" = "claimed" ]; then
          API_KEY=$(echo "$BODY" | jq -r '.apiKey // empty')
          if [ -z "$API_KEY" ]; then
            printf '\r\033[K'
            echo "Error: claimed but no apiKey returned"
            exit 1
          fi
          echo -n "$API_KEY" | sudo tee /etc/BB_API_KEY > /dev/null
          sudo chmod 644 /etc/BB_API_KEY
          restart >/dev/null 2>&1 || true
          printf '\r\033[K'
          echo "$BANNER_LINKED"
          exit 0
        fi
      done

      printf '\r\033[K'
      echo "$BANNER_TIMEOUT"
      echo "  No confirmation in 10 minutes. Run 'login' again."
      exit 1
    '';
  };

  calibrate = pkgs.stdenv.mkDerivation {
    name = "calibrate";
    version = "1.0";
    inherit src;
    nativeBuildInputs = [ makeWrapper ];
    installPhase = ''
      mkdir -p $out/bin
      ln -s "$src/calibrate.py" "$out/bin/calibrate.py"

      makeWrapper ${python}/bin/python3 $out/bin/calibrate \
        --add-flags "$out/bin/calibrate.py" \
        --set PATH ${runtimePath}
    '';
  };

  list = pkgs.stdenv.mkDerivation {
    name = "list";
    version = "1.0";
    inherit src;
    nativeBuildInputs = [ makeWrapper ];
    installPhase = ''
      mkdir -p $out/bin
      ln -s "$src/list.py" "$out/bin/list.py"

      makeWrapper ${python}/bin/python3 $out/bin/list \
        --add-flags "$out/bin/list.py" \
        --set PATH ${runtimePath}
    '';
  };

logs = pkgs.writeShellApplication {
  name = "logs";
  text = ''
    PINK="\033[1;35m"
    RED="\033[31m"
    RESET="\033[0m"

    highlight_errors() {
      awk -v red="$RED" -v reset="$RESET" '
        /Traceback|Error|Exception|KeyboardInterrupt|No such file or directory|fail|Failed/ {
          print red $0 reset;
          next;
        }
        { print }
      '
    }

    if [ "$#" -eq 0 ]; then
      # No argument provided - show all log files
      for f in /dev/shm/*.log; do
        echo -e "$PINK========== LOG FILE: $f ==========$RESET"
        head -50 "$f" | highlight_errors
        echo ""
      done
    else
      DAEMON_NAME="$1"
      
      # First try to find daemon log files
      FOUND_LOGS=false
      for f in /dev/shm/*"$DAEMON_NAME"*.log; do
        if [ -f "$f" ]; then
          echo -e "$PINK========== LOG FILE: $f ==========$RESET"
          highlight_errors < "$f"
          echo ""
          FOUND_LOGS=true
        fi
      done
      
      # If no daemon logs found, try systemd service
      if [ "$FOUND_LOGS" = false ]; then
        echo -e "$PINK========== SYSTEMD SERVICE: $DAEMON_NAME.service ==========$RESET"
        if sudo journalctl -u "$DAEMON_NAME.service" --no-pager -l 2>/dev/null | highlight_errors; then
          FOUND_LOGS=true
        fi
      fi
      
      if [ "$FOUND_LOGS" = false ]; then
        echo -e "$RED" "[ERROR] No logs found for: $DAEMON_NAME" "$RESET"
        exit 1
      fi
    fi
  '';
};

restart = pkgs.writeShellApplication {
  name = "restart";
  text = ''
    set -eu
    if [ "$#" -eq 0 ]; then
      rebuild
      find ${daemonsPath} -name '.stopped' -delete
      sudo systemctl kill -s SIGKILL manager
      sudo systemctl kill -s SIGKILL app_manager
      find /tmp -maxdepth 1 \( -name '*_lock' -o -name '*.log' \) -exec rm -f {} +
      rm -f /dev/shm/*.state /dev/shm/*.status /dev/shm/*.log /dev/shm/*.orientation /dev/shm/*.data /dev/shm/*.raw /dev/shm/*.ctrl
      sudo systemctl restart manager
      sudo systemctl restart app_manager
      echo "Restarted all daemons!"
    else
      for arg in "$@"; do
        rm -f "${daemonsPath}/$arg/.stopped"
        pkill -9 -f "python daemon.py $arg" || true
      done
      sleep 0.2
      for arg in "$@"; do
        rm -f /dev/shm/"''${arg}".* /dev/shm/"''${arg}"_lock
      done
      echo "Restarted daemons: $* (manager will auto-restart them)"
    fi
  '';
};

stop = pkgs.writeShellApplication {
  name = "stop";
  text = ''
    set -eu
    if [ "$#" -eq 0 ]; then
      sudo systemctl kill -s SIGKILL manager
      sudo systemctl kill -s SIGKILL app_manager
      find /tmp -maxdepth 1 \( -name '*_lock' -o -name '*.log' \) -exec rm -f {} +
      rm -f /dev/shm/*.state /dev/shm/*.status /dev/shm/*.log /dev/shm/*.orientation /dev/shm/*.data /dev/shm/*.raw /dev/shm/*.ctrl
      echo "Stopped!"
    else
      for arg in "$@"; do
        touch "${daemonsPath}/$arg/.stopped"
        pkill -f "python daemon.py $arg" || true
        rm -f /dev/shm/"''${arg}".* /dev/shm/"''${arg}"_lock
      done
      echo "Stopped daemons: $*"
    fi
  '';
};

constants = pkgs.writeShellApplication {
  name = "constants";
  text = ''
  ${python}/bin/python3 - "$@" <<'PY'
GREEN = "\033[1;32m"
RESET = "\033[0m"
import sys, inspect, difflib
sys.path.insert(0, '${bbosRoot}')
from bbos import all_configs
d = all_configs()
if len(sys.argv) > 1:
    for s in sys.argv[1:]:
        cfg = d.get(s, None)
        if cfg is None:
            print(f"Config '{s}' not found! Maybe: {difflib.get_close_matches(s, d.keys())}")
            continue
        print(f"{GREEN}{s} @ {inspect.getfile(cfg)}{RESET}")
        for k, v in cfg.__dict__.items():
            if not (k.startswith('__') and k.endswith('__')):
                print(f"  {k} = {v}")
else:
    for name, cfg in d.items():
        print(f"{GREEN}{name} @ {inspect.getfile(cfg)}{RESET}")
        for k, v in cfg.__dict__.items():
            if not (k.startswith('__') and k.endswith('__')):
                print(f"  {k} = {v}")
PY
  '';
};

types = pkgs.writeShellApplication {
  name = "types";
  text = ''
  ${python}/bin/python3 - "$@" <<'PY'
CYAN = "\033[36m"
RESET = "\033[0m"
import sys, difflib
sys.path.insert(0, '${bbosRoot}')
from bbos import all_types
d = all_types()
if len(sys.argv) > 1:
    for s in sys.argv[1:]:
        t = d.get(s, None)
        if t is None:
            print(f"Type '{s}' not found! Maybe: {difflib.get_close_matches(s, d.keys())}")
            continue
        print(f"{CYAN}{s}{RESET}")
        for field in t:
            print(f"  {field}")
else:
    for name, fields in d.items():
        field_names = ", ".join(f[0] for f in fields if f[0] != "timestamp")
        print(f"{CYAN}{name}{RESET}  [{field_names}]")
PY
  '';
};

disable = pkgs.writeShellApplication {
  name = "deactivate";
  text = ''
    set -eu
    if [ "$#" -eq 0 ]; then
      echo "Usage: deactivate <daemon>"
      exit 1
    fi
    DAEMON_NAME="$1"
    DAEMON_DIR="${daemonsPath}/$DAEMON_NAME"

    if [ -f "$DAEMON_DIR/.disabled" ]; then
      echo "Error: $DAEMON_NAME is already deactivated"
      exit 1
    fi

    if [ ! -d "$DAEMON_DIR" ]; then
      echo "Error: daemon directory not found: $DAEMON_DIR"
      exit 1
    fi

    touch "$DAEMON_DIR/.disabled"
    rm -f "$DAEMON_DIR/.stopped"
    pkill -f "python daemon.py $DAEMON_NAME" || true
    echo "Deactivated $DAEMON_NAME"
  '';
};

activate = pkgs.writeShellApplication {
  name = "activate";
  text = ''
    set -eu
    if [ "$#" -eq 0 ]; then
      echo "Usage: activate <daemon>"
      exit 1
    fi
    DAEMON_NAME="$1"
    DAEMON_DIR="${daemonsPath}/$DAEMON_NAME"

    if [ ! -f "$DAEMON_DIR/.disabled" ]; then
      echo "Error: $DAEMON_NAME is not deactivated"
      exit 1
    fi

    rm "$DAEMON_DIR/.disabled"
    echo "Activated $DAEMON_NAME (manager will auto-start it)"
  '';
};

clip = pkgs.writeShellApplication {
  name = "clip";
  text = ''
  ${python}/bin/python3 - "$@" <<'PY'
import sys, json, time, re
sys.path.insert(0, '${bbosRoot}')
from bbos import Config
from pathlib import Path
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: clip <description>")
    print("Example: clip 'left motor made grinding noise about 2 min ago'")
    sys.exit(1)

desc = " ".join(sys.argv[1:])
cfg = Config("logging")
clips_dir = Path(cfg.clip_folder)
clips_dir.mkdir(parents=True, exist_ok=True)

now = time.time()
event_time = now
match = re.search(r'(?:last|about|~?)\s*(\d+)\s*(sec|second|min|minute|hour|hr)s?\s*(?:ago)?', desc, re.IGNORECASE)
if match:
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith('sec'):
        event_time = now - amount
    elif unit.startswith('min'):
        event_time = now - amount * 60
    elif unit.startswith('h'):
        event_time = now - amount * 3600

clip_data = {
    "clip_time": now,
    "event_time": event_time,
    "description": desc,
}

filename = f"clip_{time.strftime('%Y%m%d_%H%M%S')}.json"
path = clips_dir / filename
path.write_text(json.dumps(clip_data, indent=2) + "\n")

GREEN = "\033[1;32m"
R = "\033[0m"
clip_dt = datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
event_dt = datetime.fromtimestamp(event_time).strftime('%Y-%m-%d %H:%M:%S')
print(f"{GREEN}Clip saved:{R} {path}")
print(f"  Clipped at: {clip_dt}")
if event_time != now:
    print(f"  Event ~at:  {event_dt}")
print(f"  Note: {desc}")
PY
  '';
};


run = pkgs.writeShellApplication {
  name = "run";
  text = ''
    set -eu
    if [ "$#" -eq 0 ]; then
      echo "Usage: run <app> [args...]"
      echo ""
      echo "Available apps:"
      for f in ~/bbapps/*.py; do
        [ -f "$f" ] && echo "  $(basename "$f" .py)"
      done
      for d in ~/bbapps/*/main.py; do
        [ -f "$d" ] && echo "  $(basename "$(dirname "$d")")"
      done
      exit 1
    fi

    APP_NAME="$1"
    shift

    if [ -f ~/bbapps/"$APP_NAME".py ]; then
      exec uv run ~/bbapps/"$APP_NAME".py "$@"
    elif [ -f ~/bbapps/"$APP_NAME"/main.py ]; then
      exec uv run ~/bbapps/"$APP_NAME"/main.py "$@"
    else
      echo "Error: app not found: $APP_NAME"
      echo "Looked for: ~/bbapps/$APP_NAME.py or ~/bbapps/$APP_NAME/main.py"
      exit 1
    fi
  '';
};

help = pkgs.writeShellApplication {
  name = "help";
  text = ''
    BOLD="\033[1m"
    CYAN="\033[36m"
    RESET="\033[0m"
    echo -e "''${BOLD}bbos CLI Commands''${RESET}"
    echo ""
    echo -e "''${CYAN}restart [daemon ...]''${RESET}     Restart all daemons, or specific ones by name"
    echo -e "''${CYAN}stop [daemon ...]''${RESET}        Stop all daemons, or specific ones by name"
    echo -e "''${CYAN}logs [daemon]''${RESET}            Show log files (all or filtered by daemon name)"
    echo -e "''${CYAN}list [filter]''${RESET}            Live TUI showing all topics, readers, frequencies"
    echo -e "''${CYAN}constants [name]''${RESET}         Print all daemon configs, or a specific one"
    echo -e "''${CYAN}types [name]''${RESET}             Print all registered types, or a specific one"
    echo -e "''${CYAN}deactivate <daemon>''${RESET}  Deactivate a daemon (won't auto-start)"
    echo -e "''${CYAN}activate <daemon>''${RESET}   Re-activate a deactivated daemon"
    echo -e "''${CYAN}calibrate <daemon>''${RESET}      Calibrate and maintain a daemon"
    echo -e "''${CYAN}clip <description>''${RESET}       Save a timestamped event clip to logging data"
    echo -e "''${CYAN}run <app> [args...]''${RESET}      Run a bbapps app by name"
    echo -e "''${CYAN}rebuild''${RESET}                   Rebuild CLI commands (after editing list.py, etc.)"
    echo -e "''${CYAN}login''${RESET}                    Link this robot to your bb-cloud account"
    echo -e "''${CYAN}help''${RESET}                     Show this help message"
    echo ""
    echo -e "''${BOLD}Running Apps''${RESET}"
    echo -e "  uv run ~/bbapps/<app>.py        Run a single-file app"
    echo -e "  uv run ~/bbapps/<app>/main.py   Run a multi-file app"
    echo ""
    echo -e "''${BOLD}Auto-starting Apps''${RESET}"
    echo -e "  Add app names to ~/bbapps/.autostart (one per line)."
    echo -e "  Apps listed there start automatically on boot."
    echo -e "  Optional args after the name:  my_app --flag value"
    echo ""
    echo -e "''${BOLD}Daemons''${RESET}  ${daemonsPath}/<name>/{daemon.py, constants.py}"
    for d in ${daemonsPath}/*/daemon.py; do
      dir=$(dirname "$d")
      name=$(basename "$dir")
      disabled=""
      if [ -f "$dir/.disabled" ]; then
        disabled=" ''${CYAN}(disabled)''${RESET}"
      fi
      echo -e "  $name$disabled  ''${CYAN}$dir''${RESET}"
    done
  '';
};

rebuild = pkgs.writeShellApplication {
  name = "rebuild";
  text = ''
    set -eu
    OLD_RESULT=$(readlink ${daemonsPath}/manager/result 2>/dev/null || echo "")
    nix-build ${daemonsPath}/manager -o ${daemonsPath}/manager/result --argstr daemonsPath ${daemonsPath} --argstr bbosRoot ${bbosRoot} > /dev/null
    NEW_RESULT=$(readlink ${daemonsPath}/manager/result)
    if [ "$OLD_RESULT" != "$NEW_RESULT" ]; then
      echo "Rebuilt: $NEW_RESULT"
    else
      echo "No changes"
    fi
  '';
};

in pkgs.buildEnv {
  name = "manager";
  paths = [ manager calibrate login logs restart stop list constants types disable activate clip run rebuild help];
}
