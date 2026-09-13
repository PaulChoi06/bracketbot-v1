# Shared devenv factory for bbos daemons.
# System dependencies come from pinned Nixpkgs; Python dependencies come from
# each daemon's pinned pyproject.toml and uv.lock.

{ pkgs
, python
, uvPackage
, extraPackages  ? []
, extraLdPaths   ? []
, env            ? {}
, preSyncHook    ? ""
, extraShellHook ? ""
}:

let
  lib = pkgs.lib;
  ldPaths = [ pkgs.stdenv.cc.cc.lib pkgs.zlib ] ++ extraLdPaths;
  ldExports = lib.concatMapStringsSep "\n"
    (p: "export LD_LIBRARY_PATH=${p}/lib\${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
    ldPaths;
  envExports = lib.concatStringsSep "\n" (lib.mapAttrsToList
    (name: value: "export ${name}=${lib.escapeShellArg (toString value)}")
    env);
in
{
  languages.python = {
    enable = true;
    package = python;
    uv = {
      enable = true;
      package = uvPackage;
      sync.enable = false;
    };
  };

  packages = [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.util-linux
  ] ++ extraPackages;

  enterShell = ''
    ${ldExports}
    ${envExports}

    export UV_PYTHON=${python}/bin/python3
    export UV_PYTHON_DOWNLOADS=never
    export UV_LINK_MODE=hardlink
    export UV_PROJECT_ENVIRONMENT=.venv
    export UV_PREVIEW=1

    # Jetson system kernel headers must be found before nix linux-headers.
    _SYS_HDR="$(pwd)/.devenv/sys-headers"
    mkdir -p "$_SYS_HDR"
    ln -sfn /usr/include/linux "$_SYS_HDR/linux"
    export NIX_CFLAGS_COMPILE="-isystem $_SYS_HDR ''${NIX_CFLAGS_COMPILE:-}"

    ${preSyncHook}

    if [ -e .venv ] && [ ! -d .venv ]; then
      rm -f .venv
    fi
    _bbos_sync_key() {
      if [ -f uv.lock ]; then
        sha256sum pyproject.toml uv.lock | sha256sum | awk '{print $1}'
      else
        sha256sum pyproject.toml | sha256sum | awk '{print $1}'
      fi
    }
    _cleanup_bbos_daemon_editables() {
      if [ -d .venv/lib ]; then
        find .venv/lib -path '*/site-packages' -type d -print | while IFS= read -r _SP; do
          rm -rf "$_SP"/bbos_daemon_*.dist-info "$_SP"/__editable__.bbos_daemon_*.pth "$_SP"/__editable___bbos_daemon_*_finder.py
        done
      fi
    }
    _cleanup_bbos_daemon_editables

    _SYNC_KEY_FILE=.venv/.bbos-sync-key
    _SYNC_KEY=$(_bbos_sync_key)
    if [ ! -d .venv/bin ] || [ ! -f "$_SYNC_KEY_FILE" ] || [ "$(cat "$_SYNC_KEY_FILE" 2>/dev/null)" != "$_SYNC_KEY" ]; then
      export _SYNC_KEY_FILE
      flock /tmp/bbos-uv-sync.lock bash -c '
        set -e
        env -u VIRTUAL_ENV uv sync --no-install-project
        _NEW_SYNC_KEY=$(if [ -f uv.lock ]; then sha256sum pyproject.toml uv.lock; else sha256sum pyproject.toml; fi | sha256sum | awk "{print \$1}")
        mkdir -p .venv
        printf "%s\n" "$_NEW_SYNC_KEY" > "$_SYNC_KEY_FILE"
      ' || exit 1
    fi
    if [ ! -d .venv/bin ]; then
      echo "ERROR: uv sync did not create .venv for $(pwd)"
      exit 1
    fi

    _SP=$(find .venv/lib -path '*/site-packages' -type d -print -quit)
    if [ -z "$_SP" ]; then
      echo "ERROR: no site-packages directory found in .venv for $(pwd)"
      exit 1
    fi
    _BBOS_ROOT=$(cd ../../.. && pwd)
    if [ ! -f "$_SP/bbos.pth" ] || [ "$(cat "$_SP/bbos.pth" 2>/dev/null)" != "$_BBOS_ROOT" ]; then
      rm -rf "$_SP/bbos" "$_SP"/bbos-*.dist-info "$_SP"/__editable__.bbos-*.pth "$_SP"/__editable___bbos_*_finder.py
      echo "$_BBOS_ROOT" > "$_SP/bbos.pth"
    fi
    _cleanup_bbos_daemon_editables

    for _CMEEL_LIB in .venv/lib/python*/site-packages/cmeel.prefix/lib; do
      if [ -d "$_CMEEL_LIB" ]; then
        export LD_LIBRARY_PATH=$(cd "$_CMEEL_LIB" && pwd)''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
      fi
    done

    unset _SP _BBOS_ROOT _CMEEL_LIB
    unset _SYNC_KEY _SYNC_KEY_FILE
    unset -f _cleanup_bbos_daemon_editables _bbos_sync_key
    source .venv/bin/activate

    ${extraShellHook}
  '';
}
