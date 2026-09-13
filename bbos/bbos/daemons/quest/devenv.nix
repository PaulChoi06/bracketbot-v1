{ ... }:

let
  pins = import ../pinned-nixpkgs.nix {};
  pkgs = pins.daemon;
in
import ../common-devenv.nix {
  inherit pkgs;
  python = pkgs.python312;
  uvPackage = pins.uv.uv;
  extraPackages = with pkgs; [
    ffmpeg libopus libvpx openssl libGL glibc glibc.dev
  ];
  extraLdPaths = with pkgs; [
    libGL openssl.out libopus libvpx
  ];
  extraShellHook = ''
    if [ ! -f key.pem ] || [ ! -f cert.pem ]; then
      openssl genrsa -out key.pem 2048
      openssl req -new -key key.pem -out cert.csr -subj "/CN=localhost"
      openssl x509 -req -in cert.csr -signkey key.pem -out cert.pem -days 365
    fi
  '';
}
