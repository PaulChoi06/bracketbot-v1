{ ... }:

let
  pins = import ../pinned-nixpkgs.nix {};
  pkgs = pins.daemon;
  cuda = import ../jetson-cuda.nix;
in
import ../common-devenv.nix {
  inherit pkgs;
  python = pkgs.python311;
  uvPackage = pins.uv.uv;

  # bbslam.so needs libcudart/libcublas (cuda targets dir) and libnvinfer,
  # which sits next to the system libc: it gets a symlink dir of its own so the
  # nix loader never sees /usr/lib/aarch64-linux-gnu itself.
  extraShellHook = ''
    _TRT_COMPAT=.devenv/trt-compat
    mkdir -p "$_TRT_COMPAT"
    ln -sf /usr/lib/aarch64-linux-gnu/libnvinfer.so.10* "$_TRT_COMPAT"/
    # the L4T jpeg library reaches EGL through nvbufsurface; the dispatcher
    # lives beside the system libc, so it gets symlinked in like libnvinfer.
    ln -sf /usr/lib/aarch64-linux-gnu/libEGL.so.1* "$_TRT_COMPAT"/
    ln -sf /usr/lib/aarch64-linux-gnu/libGLdispatch.so.0* "$_TRT_COMPAT"/
    ln -sf /usr/lib/aarch64-linux-gnu/libjpeg.so.8* "$_TRT_COMPAT"/
    export LD_LIBRARY_PATH="$(pwd)/$_TRT_COMPAT:${cuda.jetsonLibPath}:${cuda.systemNvidiaPath}:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset _TRT_COMPAT
  '';
}
