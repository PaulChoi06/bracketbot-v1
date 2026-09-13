# Jetson Devenv Shared Configurations

Reusable Nix configurations for Jetson Orin Nano (JetPack 6, CUDA 12.6) that eliminate duplication across daemons.

## Files

- **`jetson-cuda.nix`** - CUDA 12.6 configuration (paths, env vars, compiler settings)
- **`jetson-torch.nix`** - PyTorch 2.8.0 as a Nix package (stored once, reused everywhere)

## What This Solves

### Before
- Each daemon's venv had its own 705MB copy of PyTorch
- CUDA paths duplicated across every daemon environment
- Inconsistent versions across daemons

### After
- **PyTorch**: Built once as Nix derivation → stored in `/nix/store`
- **Saves 705MB per daemon** that uses PyTorch
- **CUDA config**: Defined once, imported everywhere
- **Consistent versions**: daemon Nixpkgs pins, `devenv.lock`, `pyproject.toml`, and `uv.lock` pin the environment

## Usage

### For CUDA-only daemons

```nix
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
  env = {
    inherit (cuda.envVars) CUDA_PATH CUDA_HOME CUDACXX;
  };
  preSyncHook = ''
    export PATH="${cuda.binPath}:$PATH"
    export LD_LIBRARY_PATH="${cuda.libPath}:${cuda.jetsonLibPath}:$LD_LIBRARY_PATH"
  '';
}
```

### For PyTorch + CUDA daemons

```nix
{ ... }:

let
  pins = import ../pinned-nixpkgs.nix {};
  pkgs = pins.daemon;
  cuda = import ../jetson-cuda.nix;
  jetsonTorch = import ../jetson-torch.nix {
    inherit pkgs;
    python = pkgs.python310;
  };
in
import ../common-devenv.nix {
  inherit pkgs;
  python = pkgs.python310;
  uvPackage = pins.uv.uv;
  extraPackages = [
    jetsonTorch.torch
  ];
  env = {
    inherit (cuda.envVars) CUDA_PATH CUDA_HOME CUDACXX;
  };
}
```

## Current Versions

- **CUDA**: 12.6 (JetPack 6)
- **PyTorch**: 2.8.0 (from Jetson AI Lab)
- **Python**: 3.10

## Daemons Using These

- **jetson-cuda.nix**: `mapping/`, `slam/`
- **jetson-torch.nix**: `segmenter/`, `slam/`

## How It Works

1. **First `devenv shell` entry**: PyTorch wheel is downloaded, patched, and stored in `/nix/store`
2. **Subsequent entries**: Nix reuses the cached package (instant!)
3. **venv creation**: `uv sync --locked` creates the daemon `.venv`
4. **Dependencies**: Python deps are resolved from each daemon's pinned `uv.lock`
5. **Result**: 705MB torch shared, only ~20MB deps duplicated

## Disk Space Savings

| Daemon | Before | After | Saved |
|--------|--------|-------|-------|
| segmenter | 705MB | 0MB (shared) | 705MB |
| slam | 705MB | 0MB (shared) | 705MB |
| **Total** | **1.4GB** | **705MB** | **705MB** |

And more savings as additional PyTorch-using daemons are added!

## Updating Versions

### Update CUDA version
Edit `jetson-cuda.nix`:
```nix
basePath = "/usr/local/cuda-12.7";  # Change here
```

### Update PyTorch version
Edit `jetson-torch.nix`:
```nix
version = "2.9.0";  # Change version
url = "https://...";  # Update URL
sha256 = "sha256-...";  # Run nix-prefetch-url to get new hash
```

## Benefits

✅ **Space Efficient**: 705MB saved per daemon  
✅ **DRY Principle**: No config duplication  
✅ **Fast Builds**: Nix caching makes subsequent builds instant  
✅ **Reproducible**: Exact versions pinned with SHA256 hashes  
✅ **Maintainable**: Update versions in one place  
