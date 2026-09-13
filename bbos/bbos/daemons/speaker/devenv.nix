{ ... }:

let
  pins = import ../pinned-nixpkgs.nix {};
  pkgs = pins.daemon;
in
import ../common-devenv.nix {
  inherit pkgs;
  python = pkgs.python311;
  uvPackage = pins.uv.uv;
  extraPackages = [ pkgs.pulseaudio ];
}
