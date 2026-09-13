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
    libGL xorg.libX11 gcc14
    cmake ninja gcc-arm-embedded dfu-util git libusb1 usbutils
  ];
  extraLdPaths = with pkgs; [
    libGL xorg.libX11 libusb1
  ];
}
