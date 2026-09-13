# Reusable PyTorch package for Jetson Orin Nano (JetPack 6, CUDA 12.6)
# Import this in devenv.nix with: jetsonTorch = import ./jetson-torch.nix { inherit pkgs python; };
# where python is the Python version you want to use (e.g., python310)
{ pkgs, python }:

let
  # Import CUDA configuration
  cuda = import ./jetson-cuda.nix;

  # PyTorch for Jetson from jetson-ai-lab.io
  # Dependencies (typing-extensions, etc.) are resolved by each daemon's uv.lock.
  torch = python.pkgs.buildPythonPackage rec {
    pname = "torch";
    version = "2.8.0";
    format = "wheel";
    
    src = pkgs.fetchurl {
      url = "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl";
      sha256 = "sha256-YqG+7p8vFHB2qXTSlCyQBgwSdxyUdAgwMnyucFsllfw=";
    };
    
    nativeBuildInputs = with pkgs; [ 
      autoPatchelfHook
      patchelf
    ];
    
    buildInputs = with pkgs; [
      stdenv.cc.cc.lib
      zlib
      libGL
      glibc
    ];
    
    # Convert RPATH to RUNPATH so LD_LIBRARY_PATH takes precedence
    # This allows devenv.nix to provide CUDA libs via LD_LIBRARY_PATH at runtime
    postFixup = ''
      for lib in $out/lib/python3.10/site-packages/torch/lib/*.so*; do
        if [ -f "$lib" ] && ! [ -L "$lib" ]; then
          echo "Converting RPATH to RUNPATH for $lib"
          # --force-rpath with --enable-new-dtags converts RPATH to RUNPATH
          # RUNPATH allows LD_LIBRARY_PATH to be searched
          patchelf --force-rpath --enable-new-dtags "$lib" 2>/dev/null || true
        fi
      done
    '';
    
    # No propagatedBuildInputs - let each daemon's uv environment handle dependencies.
    propagatedBuildInputs = [ ];

    # Skip dependency checks while installing the wheel into the Nix package.
    pipInstallFlags = [ "--no-deps" ];
    
    # Don't check for missing dependencies or run tests
    doCheck = false;
    autoPatchelfIgnoreMissingDeps = true;
    pythonImportsCheck = [ ];
    dontStrip = true;
    
    # Skip phases
    dontConfigure = true;
    dontBuild = true;
  };

  # TorchVision for Jetson from jetson-ai-lab.io
  torchvision = python.pkgs.buildPythonPackage rec {
    pname = "torchvision";
    version = "0.23.0";
    format = "wheel";
    
    src = pkgs.fetchurl {
      url = "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl";
      sha256 = "sha256-kHxMGTN4lkXrsg3ZGB1A+GR5eOa9MAhq57Af67k30tE=";
    };
    
    nativeBuildInputs = with pkgs; [ 
      autoPatchelfHook
      patchelf
    ];
    
    buildInputs = with pkgs; [
      stdenv.cc.cc.lib
    ];
    
    # Convert RPATH to RUNPATH so LD_LIBRARY_PATH takes precedence
    postFixup = ''
      for lib in $out/lib/python3.10/site-packages/torchvision/*.so*; do
        if [ -f "$lib" ] && ! [ -L "$lib" ]; then
          echo "Converting RPATH to RUNPATH for $lib"
          patchelf --force-rpath --enable-new-dtags "$lib" 2>/dev/null || true
        fi
      done
    '';
    
    # torchvision depends on torch and pillow
    propagatedBuildInputs = [ 
      torch
      python.pkgs.pillow
    ];
    
    # Skip dependency checks while installing the wheel into the Nix package.
    pipInstallFlags = [ "--no-deps" ];
    
    # Don't check for missing dependencies or run tests
    doCheck = false;
    autoPatchelfIgnoreMissingDeps = true;
    pythonImportsCheck = [ ];
    dontStrip = true;
    
    # Skip phases
    dontConfigure = true;
    dontBuild = true;
  };

  # TorchAudio for Jetson from jetson-ai-lab.io
  torchaudio = python.pkgs.buildPythonPackage rec {
    pname = "torchaudio";
    version = "2.8.0";
    format = "wheel";
    
    src = pkgs.fetchurl {
      url = "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/81a/775c8af36ac85/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl";
      sha256 = "sha256-gad1yK82rIWfs/ShsvZi1fzyhKg1trtO2NCCemqpwLc=";
    };
    
    nativeBuildInputs = with pkgs; [ 
      autoPatchelfHook
      patchelf
    ];
    
    buildInputs = with pkgs; [
      stdenv.cc.cc.lib
    ];
    
    # Convert RPATH to RUNPATH so LD_LIBRARY_PATH takes precedence
    postFixup = ''
      for lib in $out/lib/python3.10/site-packages/torchaudio/*.so*; do
        if [ -f "$lib" ] && ! [ -L "$lib" ]; then
          echo "Converting RPATH to RUNPATH for $lib"
          patchelf --force-rpath --enable-new-dtags "$lib" 2>/dev/null || true
        fi
      done
    '';
    
    # torchaudio depends on torch
    propagatedBuildInputs = [ 
      torch
    ];
    
    # Skip dependency checks while installing the wheel into the Nix package.
    pipInstallFlags = [ "--no-deps" ];
    
    # Don't check for missing dependencies or run tests
    doCheck = false;
    autoPatchelfIgnoreMissingDeps = true;
    pythonImportsCheck = [ ];
    dontStrip = true;
    
    # Skip phases
    dontConfigure = true;
    dontBuild = true;
  };

in
{
  inherit torch torchvision torchaudio;
}
