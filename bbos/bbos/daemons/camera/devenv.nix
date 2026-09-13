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
    libjpeg_turbo.out ffmpeg libGL glib glibc xorg.libX11 xorg.libxcb glibc.dev
  ];
  extraLdPaths = with pkgs; [
    libGL glib.out xorg.libX11 xorg.libxcb libjpeg_turbo.out
  ];
}
