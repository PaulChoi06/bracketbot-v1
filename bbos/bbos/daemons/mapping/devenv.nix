{ ... }:

let
  pins = import ../pinned-nixpkgs.nix {};
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/4d2b37a84fad1091b9de401eb450aae66f1a741e.tar.gz";
    sha256 = "11w3wn2yjhaa5pv20gbfbirvjq6i3m7pqrq2msf0g7cv44vijwgw";
  }) { system = "aarch64-linux"; };
  cuda = import ../jetson-cuda.nix;
  cudaLdPath = pkgs.lib.concatStringsSep ":" [
    cuda.cudaCompat
    cuda.systemNvidiaPath
    cuda.libPath
    cuda.jetsonLibPath
  ];
in
import ../common-devenv.nix {
  inherit pkgs;
  python = pkgs.python310;
  uvPackage = pins.uv.uv;

  extraPackages = with pkgs; [
    gcc gnumake cmake pkg-config git git-lfs file
    gcc12
    python310.pkgs.wheel
    python310.pkgs.setuptools
    libGL glibc glibc.dev libdrm xorg.libX11 xorg.libxcb wayland
  ];

  extraLdPaths = with pkgs; [
    gcc12.cc.lib libGL xorg.libX11 xorg.libxcb wayland libdrm
  ];

  env = cuda.envVars;

  extraShellHook = ''
    _BBOS_CUDA_LIBS=.devenv/cuda-runtime-libs
    mkdir -p "$_BBOS_CUDA_LIBS"
    for _LIB in /usr/lib/aarch64-linux-gnu/libcudnn*.so* /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so* /usr/lib/aarch64-linux-gnu/libnvinfer.so.10*; do
      if [ -e "$_LIB" ]; then
        ln -sf "$_LIB" "$_BBOS_CUDA_LIBS"/
      fi
    done

    export PATH="${cuda.binPath}:$PATH"
    export LD_LIBRARY_PATH="$(pwd)/$_BBOS_CUDA_LIBS:${cudaLdPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CPATH="${cuda.includePath}''${CPATH:+:$CPATH}"
    export LIBRARY_PATH="${cuda.compilerPaths.LIBRARY_PATH}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
    unset _BBOS_CUDA_LIBS _LIB
  '';
}
