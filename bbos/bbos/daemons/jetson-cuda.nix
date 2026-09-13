# Reusable CUDA configuration for Jetson Orin Nano (JetPack 6, CUDA 12.6)
# Import this in devenv.nix with: cuda = import ./jetson-cuda.nix;
rec {
  # Base CUDA path
  basePath = "/usr/local/cuda-12.6";
  
  # CUDA directories
  binPath = "${basePath}/bin";
  libPath = "${basePath}/lib64";
  # Jetson-specific CUDA library path
  jetsonLibPath = "${basePath}/targets/aarch64-linux/lib";
  includePath = "${basePath}/include";
  pkgConfigPath = "${basePath}/targets/aarch64-linux/lib/pkgconfig";
  
  # System NVIDIA library paths (for runtime libraries)
  systemNvidiaPath = "/usr/lib/aarch64-linux-gnu/nvidia";
  systemLibPath = "/usr/lib/aarch64-linux-gnu";
  
  # Additional paths for CUDA compatibility
  cudaCompat = "/usr/local/cuda/compat";
  
  # CUDA executables
  nvcc = "${binPath}/nvcc";
  
  # Environment variables
  envVars = {
    CUDA_PATH = basePath;
    CUDA_HOME = basePath;
    CUDACXX = nvcc;
    CUDAHOSTCXX = "g++";
    PKG_CONFIG_PATH = pkgConfigPath;
    # PyTorch CUDA settings
    TORCH_CUDA_ARCH_LIST = "7.2;8.7";  # Jetson Orin Nano compute capability
    FORCE_CUDA = "1";
    # Add CUDA visible devices
    CUDA_VISIBLE_DEVICES = "0";
  };
  
  # Library paths for runtime - order matters!
  ldLibraryPaths = [
    # CUDA compatibility libraries first
    cudaCompat
    # System NVIDIA libraries (contains actual runtime libs)
    systemNvidiaPath
    systemLibPath
    # CUDA 12.6 libraries
    libPath
    jetsonLibPath
    # Additional system paths
    "/usr/local/lib"
    "/lib/aarch64-linux-gnu"
  ];
  
  # Compiler paths
  compilerPaths = {
    CPATH = includePath;
    LIBRARY_PATH = "${libPath}:${jetsonLibPath}:${systemLibPath}";
  };
}
