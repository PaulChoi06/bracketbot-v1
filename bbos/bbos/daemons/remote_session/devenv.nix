{ ... }:

let
  pins = import ../pinned-nixpkgs.nix {};
  pkgs = pins.daemon;
in
import ../common-devenv.nix {
  inherit pkgs;
  python = pkgs.python311;
  uvPackage = pins.uv.uv;
  extraPackages = with pkgs; [
    ffmpeg libopus libvpx openssl libGL glib glibc glibc.dev libjpeg_turbo.out xorg.libxcb
  ];
  extraLdPaths = with pkgs; [
    libGL glib.out openssl.out libopus libvpx libjpeg_turbo.out xorg.libxcb
  ];
}
