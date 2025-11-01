# AOTI PT2 Model Export and Inference

This directory contains scripts to export F<sup>3</sup> (Fast Feature Fields), DepthAnythingV2, and FlowHead models to PyTorch 2.x AOTI (Ahead-Of-Time Inductor) `.pt2` format and run inference benchmarks in both Python and C++.

**Tested on:**
- PyTorch 2.8 with CUDA 12.6 on NVIDIA RTX 4090
- PyTorch 2.8 with CUDA 12.6 on NVIDIA Jetson Orin (JP 6.2, Tegra 36.4)

> **Note:** If you need help setting up a Docker environment for Jetson Orin, please [raise an issue](https://github.com/grasp-lyrl/fast-feature-fields/issues) on the repository.


## Table of Contents
- [AOTI PT2 Model Export and Inference](#aoti-pt2-model-export-and-inference)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
    - [Python Environment](#python-environment)
    - [C++ Environment](#c-environment)
  - [Model Export](#model-export)
    - [Export F3 Model](#export-f3-model)
    - [Export DepthAnythingV2 Model](#export-depthanythingv2-model)
    - [Export FlowHead Model](#export-flowhead-model)
  - [Python Inference](#python-inference)
    - [Run F3 Only](#run-f3-only)
    - [Run Full Pipeline (F3 + Depth + Flow)](#run-full-pipeline-f3--depth--flow)
  - [C++ Inference](#c-inference)
    - [Build the C++ Test Executable](#build-the-c-test-executable)
    - [Run F3 Only](#run-f3-only-1)
    - [Run Full Pipeline (F3 + Depth + Flow)](#run-full-pipeline-f3--depth--flow-1)
  - [File Structure](#file-structure)
  - [Performance Notes](#performance-notes)
    - [Optimization Tips](#optimization-tips)
    - [Inference Speed Benchmarks](#inference-speed-benchmarks)
  - [Citation](#citation)

---

## Overview

The AOTI PT2 format allows you to compile PyTorch models ahead-of-time into optimized binaries that can be deployed without the full F3 package dependencies. This is particularly useful for:
- Deploying models on edge devices (e.g., Jetson Orin)
- Reducing inference latency
- Minimizing deployment package size
- Running models in C++ environments and subsequently integrating with ROS2

---

## Prerequisites

### Python Environment
As long as you have installed F<sup>3</sup> following the instructions in the [main README](../README.md), you have all the required Python dependencies for model export and inference.

### C++ Environment
- CMake >= 3.18
- C++17 compatible compiler (GCC 11.4+, Clang 5+, MSVC 2017+)
- LibTorch (automatically detected from your Python PyTorch installation)

---

## Model Export

### Export F3 Model

The F3 model is the core event-based feature field network. Export it using:

```bash
python export_f3_to_pt2.py \
    --model-name 1280x720x20_patchff_ds1_small \
    --output-name f3_aoti.pt2 \
    --build-hashed-feats \
    --n-events 200000 \
    --warmup-runs 20 \
    --runs 200
```

**Key Arguments:**
- `--model-name`: Model configuration (default: `1280x720x20_patchff_ds1_small`)
- `--output-name`: Output `.pt2` filename (default: `f3_aoti.pt2`)
- `--build-hashed-feats`: Build multi-resolution hash encoding features (recommended precomputation optimization)
- `--n-events`: Number of events for testing (default: 200000)
- `--checkpoint PATH`: Path to pretrained checkpoint `.pth` file (optional)
- `--variable-events`: Compile for variable event counts (enables dynamic shapes)
- `--warmup-runs`: Warmup iterations before benchmarking (default: 20)
- `--runs`: Number of inference runs for benchmarking (default: 200)

**Example with checkpoint:**
```bash
python export_f3_to_pt2.py \
    --model-name 1280x720x20_patchff_ds1_small \
    --checkpoint /path/to/f3_checkpoint.pth \
    --output-name f3_aoti.pt2 \
    --build-hashed-feats
```

### Export DepthAnythingV2 Model

The DepthAnythingV2 model processes F3 features to predict depth:

```bash
python export_dav2_to_pt2.py \
    --encoder vitb \
    --input-channels 32 \
    --height 238 \
    --width 308 \
    --output-name dav2_aoti.pt2 \
    --autocast bfloat16 \
    --warmup-runs 20 \
    --runs 200
```

**Key Arguments:**
- `--encoder`: Vision Transformer encoder size: `vits`, `vitb`, `vitl`, `vitg` (default: `vitb`)
- `--input-channels`: Number of input feature channels from F3 (default: 32)
- `--height`: Input height (default: 238)
- `--width`: Input width (default: 308)
- `--output-name`: Output `.pt2` filename (default: `dav2_aoti.pt2`)
- `--checkpoint PATH`: Path to pretrained checkpoint `.pth` file (optional)
- `--autocast`: Use mixed precision: `float16`, `bfloat16`, or `None` (default: None)
- `--warmup-runs`: Warmup iterations (default: 20)
- `--runs`: Benchmark iterations (default: 200)

**Example with checkpoint:**
```bash
python export_dav2_to_pt2.py \
    --encoder vitb \
    --checkpoint /path/to/dav2_checkpoint.pth \
    --input-channels 32 \
    --height 238 \
    --width 308 \
    --autocast bfloat16 \
    --output-name dav2_aoti.pt2
```

### Export FlowHead Model

The FlowHead model processes F3 features to predict optical flow:

```bash
python export_flowhead_to_pt2.py \
    --input-channels 32 \
    --height 238 \
    --width 308 \
    --kernels 9 9 9 9 \
    --btlncks 2 2 2 2 \
    --dilations 1 1 1 1 \
    --output-name flowhead_aoti.pt2 \
    --autocast float16 \
    --warmup-runs 20 \
    --runs 200
```

**Key Arguments:**
- `--input-channels`: Number of input feature channels from F3 (default: 32)
- `--height`: Input height (default: 238)
- `--width`: Input width (default: 308)
- `--kernels`: Kernel sizes for decoder blocks (default: `[9, 9, 9, 9]`)
- `--btlncks`: Bottleneck factors for decoder blocks (default: `[2, 2, 2, 2]`)
- `--dilations`: Dilation rates for decoder blocks (default: `[1, 1, 1, 1]`)
- `--output-name`: Output `.pt2` filename (default: `flowhead_aoti.pt2`)
- `--checkpoint PATH`: Path to pretrained checkpoint `.pth` file (optional)
- `--autocast`: Use mixed precision: `float16`, `bfloat16`, or `None` (default: None)
- `--warmup-runs`: Warmup iterations (default: 20)
- `--runs`: Benchmark iterations (default: 200)

**Example with checkpoint:**
```bash
python export_flowhead_to_pt2.py \
    --checkpoint /path/to/flowhead_checkpoint.pth \
    --input-channels 32 \
    --height 238 \
    --width 308 \
    --autocast float16 \
    --output-name flowhead_aoti.pt2
```

---

## Python Inference

Test the exported `.pt2` models using the Python test script. This script can run F3 alone or the full pipeline with depth and/or flow prediction.

### Run F3 Only
```bash
cd _py_src
python test_exported_pt2.py \
    --f3_pt2_path ../f3_aoti.pt2 \
    --h5_file /path/to/car_urban_day_penno_small_loop_data.h5 \
    --time_ms 30000 \
    --runs 200 \
    --warmup_runs 20
```

### Run Full Pipeline (F3 + Depth + Flow)
```bash
cd _py_src
python test_exported_pt2.py \
    --f3_pt2_path ../f3_aoti.pt2 \
    --dav2_pt2_path ../dav2_aoti.pt2 \
    --flowhead_pt2_path ../flowhead_aoti.pt2 \
    --h5_file /path/to/car_urban_day_penno_small_loop_data.h5 \
    --time_ms 30000 \
    --dav2_height 238 \
    --dav2_width 308 \
    --flow_height 238 \
    --flow_width 308 \
    --runs 200 \
    --warmup_runs 20
```

**Key Arguments:**
- `--f3_pt2_path`: Path to F3 `.pt2` file (required)
- `--dav2_pt2_path`: Path to DepthAnythingV2 `.pt2` file (optional)
- `--flowhead_pt2_path`: Path to FlowHead `.pt2` file (optional)
- `--h5_file`: Path to M3ED H5 dataset file (default: `car_urban_day_penno_small_loop_data.h5`)
- `--time_ms`: Timestamp in milliseconds to extract events (default: 30000)
- `--dav2_height/--dav2_width`: Input dimensions for depth model (default: 238×308)
- `--flow_height/--flow_width`: Input dimensions for flow model (default: 238×308)
- `--runs`: Number of timed inference runs (default: 200)
- `--warmup_runs`: Number of warmup runs (default: 20)

**Output Files:**
- `events_plot.png`: Visualization of input events
- `f3_feat_pca.png`: PCA visualization of F3 features
- `depth_colored.png`: Depth prediction (if using DAV2)
- `flow_viz.png`: Optical flow visualization (if using FlowHead)
- `flow_viz_overlay.png`: Flow overlaid on events (if using FlowHead)

---

## C++ Inference

For deployment on edge devices or C++ applications, use the C++ test executable.

### Build the C++ Test Executable

```bash
cd _cpp_src
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
```

OR if you prefer using Ninja:

```bash
cd _cpp_src
mkdir -p build
cd build
cmake .. -G Ninja
ninja
```

This will create the `test_exported_pt2` executable.

### Run F3 Only
```bash
./test_exported_pt2 \
    --f3_pt2_path ../../f3_aoti.pt2 \
    --runs 200 \
    --warmup_runs 20
```

### Run Full Pipeline (F3 + Depth + Flow)
```bash
./test_exported_pt2 \
    --f3_pt2_path ../../f3_aoti.pt2 \
    --dav2_pt2_path ../../dav2_aoti.pt2 \
    --flowhead_pt2_path ../../flowhead_aoti.pt2 \
    --dav2_height 238 \
    --dav2_width 308 \
    --flow_height 238 \
    --flow_width 308 \
    --runs 200 \
    --warmup_runs 20
```

**Key Arguments:**
- `--f3_pt2_path`: Path to F3 `.pt2` file (required)
- `--dav2_pt2_path`: Path to DepthAnythingV2 `.pt2` file (optional)
- `--flowhead_pt2_path`: Path to FlowHead `.pt2` file (optional)
- `--dav2_height/--dav2_width`: Input dimensions for depth model (default: 238×308)
- `--flow_height/--flow_width`: Input dimensions for flow model (default: 238×308)
- `--runs`: Number of timed inference runs (default: 200)
- `--warmup_runs`: Number of warmup runs (default: 20)

**Note:** The C++ version uses randomly generated event data. For real event data, you'll need to implement H5 file loading in C++ (not included in the current implementation). ROS2 support will be added soon.

---

## File Structure

```
_aoti_pt2/
├── README.md                   # This file
├── export_f3_to_pt2.py         # Export F3 model to PT2
├── export_dav2_to_pt2.py       # Export DepthAnythingV2 to PT2
├── export_flowhead_to_pt2.py   # Export FlowHead to PT2
├── f3_aoti.pt2                 # Exported F3 model (generated)
├── dav2_aoti.pt2               # Exported DAV2 model (generated)
├── flowhead_aoti.pt2           # Exported FlowHead model (generated)
├── _py_src/
│   └── test_exported_pt2.py    # Python inference test script
└── _cpp_src/
    ├── CMakeLists.txt          # CMake build configuration
    ├── test_exported_pt2.cpp   # C++ inference test implementation
    └── build/                  # Build directory (generated)
```

---

## Performance Notes

### Optimization Tips
1. **Enable hashed features**: Use `--build-hashed-feats` when exporting F<sup>3</sup> for better performance (0.62 ms vs 1.13 ms on RTX 4090). This precomputes multi-resolution hash encoding features for all the event voxels.
   - **Trade-off**: Model size increases to include W×H×L×F precomputed float32 elements (where W=width, H=height, L=levels, F=feature_size), which can be quite large (example 17MB vs 563MB for HD resolution with 20 ms of time).
   - **Recommendation**: Use `--build-hashed-feats` for deployment when memory is not constrained and latency is critical
2. **Mixed precision**: Consider using `--autocast float16` for FlowHead and `--autocast bfloat16` for DAV2 on devices with Tensor Cores

### Inference Speed Benchmarks

| Platform and GPU | F<sup>3</sup> | F<sup>3</sup> + DepthAnythingV2 | F<sup>3</sup> + FlowHead |
|--------|-------------|------------------------|------------------|
| Desktop RTX 4090 | 0.62 | 2.87 | 2.18 |
| Jetson Orin (JP 6.2) | TBD | TBD | TBD |

*Note: All timings in milliseconds (ms). Benchmarks performed with PyTorch 2.8, CUDA 12.6 using F<sup>3</sup> model configuration `1280x720x20_patchff_ds1_small`. F<sup>3</sup> timings use float16 with `--build-hashed-feats` enabled, and downsampling to 240×320 with 200k events. F<sup>3</sup> + downstream task timings include feature interpolation to (238×308) and autocasting to float16/bfloat16*


---

## Citation

If you use this code, please cite:

```bibtex
@misc{das2025fastfeaturefield,
  title={Fast Feature Field ($\text{F}^3$): A Predictive Representation of Events}, 
  author={Richeek Das and Kostas Daniilidis and Pratik Chaudhari},
  year={2025},
  eprint={2509.25146},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.25146}
}
```
