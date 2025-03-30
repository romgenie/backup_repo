# Performance Optimization for AI-Scientist

## System Specifications

Optimized for:
- **Hardware**: Apple M2 Max
- **CPU**: 12 cores (12 physical, 12 logical)
- **Memory**: 32GB RAM
- **GPU**: Apple Silicon integrated GPU via Metal Performance Shaders (MPS)
- **OS**: macOS Darwin Kernel 24.3.0
- **PyTorch**: 2.6.0 with MPS backend support

## Changes Made

### GPU/MPS Configuration
- Created `config_gpu.ini` to configure MPS backend for Apple Silicon
- Configured device selection to use MPS when available
- Set memory limit to 30GB for dedicated ML workloads
- Implemented periodic cache clearing based on `empty_cache_frequency` setting

### Implementation Status

We have successfully implemented all the optimizations specified in the config_gpu.ini file into the experiment.py code:

#### Fully Implemented:
- **Device Selection**
  - Automatic detection and use of MPS on Apple Silicon
  - Fallback mechanisms for CUDA and CPU
  - Device-specific error handling

- **Memory Management**
  - Memory fraction control (via `memory_fraction` setting)
  - Periodic cache clearing (via `empty_cache_frequency` setting)
  - Pinned memory for faster transfers (via `enable_pinned_memory` setting)
  - Prefetch factor customization (via `prefetch_factor` setting)

- **Precision Settings**
  - Support for different precision levels (`default_dtype`)
  - Override to float32 for MPS stability when needed
  - Proper error handling for unsupported configurations

- **Thread Optimization**
  - Thread count configuration (via `num_threads`)
  - OpenMP settings for thread binding and optimization
  - Adaptive thread detection based on available CPU cores

- **Backend Optimizations**
  - CUDA-specific optimizations for systems with NVIDIA GPUs
  - Kernel optimization settings
  - Memory format preferences

- **Data Loading**
  - Smart path detection and multiple fallbacks
  - Optimized batch handling
  - Asynchronous data loading

## Performance Benefits

The implemented optimizations provide:

1. **Metal Performance Shaders Support**
   - Native utilization of Apple Silicon GPU architecture
   - Proper data type handling (using float32 for stability)
   - Memory management tuned for Apple's unified memory architecture

2. **Memory Optimizations**
   - Configurable memory allocation (up to 99% of GPU memory)
   - Periodic cache clearing to prevent fragmentation
   - Optimized transfers between CPU and GPU

3. **CPU Thread Management**
   - Utilization of all 12 cores available on M2 Max
   - Thread binding for better cache locality
   - Active waiting policy for reduced latency

4. **Stability Improvements**
   - Proper error handling throughout the codebase
   - Fallback mechanisms for unsupported configurations
   - Clear logging of applied settings

## Usage

The optimizations are controlled through the `config_gpu.ini` file, which provides a centralized place to adjust performance parameters. The file has been placed in the template directory to ensure it's found by the application code.

To run the optimized code:
```bash
cd templates/nanoGPT
python experiment.py --out_dir run_0
```

The code will automatically detect and apply all the optimizations based on the configuration file.

## Next Steps

Further performance improvements could include:

1. JIT compilation support when available for MPS
2. Adaptive batch size based on available memory
3. Mixed precision training with careful handling of MPS limitations
4. Benchmarking suite to compare different configuration settings