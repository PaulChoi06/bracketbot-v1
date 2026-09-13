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
  extraPackages = with pkgs; [
    libdrm xorg.libxcb wayland libGL glib glibc xorg.libX11 glibc.dev
  ];
  extraLdPaths = with pkgs; [
    libGL glib.out xorg.libX11 xorg.libxcb wayland libdrm
  ];

  # Added NVIDIA specific paths, TODO: this can probably be simplified more
  extraShellHook = ''
    _EGL_COMPAT=.devenv/tegra-egl-compat
    mkdir -p "$_EGL_COMPAT"
    ln -sf /usr/lib/aarch64-linux-gnu/libEGL.so.1* "$_EGL_COMPAT"/
    export LD_LIBRARY_PATH="$(pwd)/$_EGL_COMPAT:${cuda.systemNvidiaPath}:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:/opt/nvidia/vpi3/lib/aarch64-linux-gnu''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset _EGL_COMPAT
  '';
}
