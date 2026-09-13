{ ... }:

let
  pins = import ../pinned-nixpkgs.nix {};
  pkgs = pins.daemon;
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
  env = cuda.envVars;

  extraShellHook = ''
    _BBOS_CUDA_LIBS=.devenv/cuda-runtime-libs
    mkdir -p "$_BBOS_CUDA_LIBS"
    for _LIB in /usr/lib/aarch64-linux-gnu/libcudnn*.so*; do
      if [ -e "$_LIB" ]; then
        ln -sf "$_LIB" "$_BBOS_CUDA_LIBS"/
      fi
    done

    export PATH="${cuda.binPath}:$PATH"
    export LD_LIBRARY_PATH="$(pwd)/$_BBOS_CUDA_LIBS:${cudaLdPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset _BBOS_CUDA_LIBS _LIB
  '';
}
