{ system ? "aarch64-linux" }:

{
  daemon = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/63dacb46bf939521bdc93981b4cbb7ecb58427a0.tar.gz";
    sha256 = "sha256:1lr1h35prqkd1mkmzriwlpvxcb34kmhc9dnr48gkm8hh089hifmx";
  }) { inherit system; };

  uv = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/b3da656039dc7a6240f27b2ef8cc6a3ef3bccae7.tar.gz";
    sha256 = "sha256:1hyl221q0c2zw3m1nv8vc39dcyrvxmn4crbn13f8p2pmcmg6x2i3";
  }) { inherit system; };
}
