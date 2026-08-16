
# 360 Gaussian Splatting

<div align="center">
  
  <a href="https://www.youtube.com/watch?v=AhWHeEB8-vc">
    <img src="https://github.com/inuex35/360-gaussian-splatting/assets/129066540/25cb8760-0709-445d-a535-9885ba2786b7" width="640" alt="360 gaussian splatting with spherical render">
  </a>
  
</div>

This repository contains programs for reconstructing space using OpenSfM and Gaussian Splatting. For original repositories of OpenSfM and Gaussian Splatting, please refer to the links provided.

# Support me
This is just my personal project.
If you've enjoyed using this project and found it helpful, 
I'd be incredibly grateful if you could chip in a few bucks to help cover the costs of running the GPU server. 
You can easily do this by buying me a coffee at 
https://www.buymeacoffee.com/inuex35. 

## Environment Setup

### Cloning the Repository

Clone the repository with the following command:

```bash
git clone --recursive https://github.com/inuex35/360-gaussian-splatting
```

### Creating the Environment

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment (Python 3.14, pinned in `.python-version`).

First, sync the regular dependencies (torch, torchvision, numpy, opencv, plyfile, pyproj, etc.):

```bash
uv sync
```

`torch`/`torchvision` are pulled from the PyTorch `cu126` wheel index (see `[tool.uv.sources]` in `pyproject.toml`). If your machine has a different CUDA toolkit/driver, point `[[tool.uv.index]]` in `pyproject.toml` at the matching `https://download.pytorch.org/whl/<cuXXX>` index instead.

The CUDA extensions (`diff-gaussian-rasterization`, `simple-knn`) must then be compiled against the *same* CUDA toolkit version that built your `torch` wheel (`torch.version.cuda`), using `nvcc` from that toolkit (not necessarily the system default one):

```bash
CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH \
  uv pip install --no-build-isolation -e submodules/simple-knn -e submodules/diff-gaussian-rasterization
```

Adjust `CUDA_HOME`/`PATH` to wherever your CUDA 12.6 toolkit is installed (`nvcc --version` should report 12.6 to match `torch.version.cuda`).

Verify the install:

```bash
uv run python -c "import torch, diff_gaussian_rasterization, simple_knn._C; print(torch.__version__, torch.cuda.is_available())"
```

#### Note on the CUDA submodules

The upstream `diff-gaussian-rasterization` and `simple-knn` sources were written against an older CUDA/GCC toolchain and do not compile as-is with recent nvcc/GCC. This repo's submodules therefore point at patched forks, so `git clone --recursive` gives you a tree that builds — no manual patching required:

| Submodule | Fork | Upstream |
| --- | --- | --- |
| `submodules/diff-gaussian-rasterization` | [thehiddenwaffle/360-diff-gaussian-rasterization](https://github.com/thehiddenwaffle/360-diff-gaussian-rasterization) (`main`) | [inuex35/360-diff-gaussian-rasterization](https://github.com/inuex35/360-diff-gaussian-rasterization) |
| `submodules/simple-knn` | [thehiddenwaffle/simple-knn](https://github.com/thehiddenwaffle/simple-knn) (`patched`) | [gitlab.inria.fr/bkerbl/simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) |

The fixes carried in the forks are:

- `simple_knn.cu`: add `#include <cfloat>` (fixes `identifier "FLT_MAX" is undefined`).
- `cuda_rasterizer/rasterizer_impl.h`: add `#include <cstdint>` (fixes `identifier "uint32_t"/"uint64_t" is undefined`).
- `simple_knn/__init__.py`: added — the package shipped without one, so PEP 660 editable installs can't resolve `import simple_knn`.

### For omnigs rendering

You can use omnigs implementation. Checkout diff-gaussian-rasterization to omnigs branch.

```bash
cd submodules/diff-gaussian-rasterization
git checkout omnigs
```

### For Depth and Normal Rendering

If you use depth and normal for training, use depth_normal_render

```bash
git clone --recursive -b depth_normal_render https://github.com/inuex35/360-gaussian-splatting 
```

and 

```bash
cd 360-gaussian-splatting
git clone https://github.com/inuex35/360-dn-diff-gaussian-rasterization submodules/360-dn-diff-gaussian-rasterization
pip3 install submodules/360-dn-diff-gaussian-rasterization submodules/simple-knn plyfile pyproj openexr imageio
```

## Starting from a 360 video

The training pipeline consumes **images**, not video, so a 360 capture (e.g. RICOH THETA `.MP4`) has to be split into frames first. THETA footage is already stitched equirectangular, so no stitching step is needed — check with:

```bash
ffprobe -v error -select_streams v:0 -show_streams your_video.MP4 | grep -i projection
# projection=equirectangular
```

Extract frames at a few fps. Every frame of 30fps footage is redundant and will make reconstruction far slower; 2-3 fps is a reasonable starting point, higher if the camera moved quickly:

```bash
mkdir -p data/your_data/images
ffmpeg -i your_video.MP4 -vf fps=3 -q:v 2 data/your_data/images/%04d.jpg
```

> **Frames extracted by ffmpeg carry no EXIF**, so OpenSfM cannot infer the camera model and will silently fall back to a perspective camera, producing a wrong reconstruction. Force the spherical model with a `camera_models_overrides.json` in the dataset root (use your real frame dimensions):

```json
{
    "all": {
        "projection_type": "spherical",
        "width": 3840,
        "height": 1920
    }
}
```

After `extract_metadata`, confirm the override took effect — `camera_models.json` should report `"projection_type": "spherical"`.

A `config.yaml` in the same directory tunes the run. OpenSfM defaults to `processes: 1` and exhaustive pair matching, which is slow; for sequential video frames:

```yaml
processes: 16                      # match your core count
matching_order_neighbors: 20       # frames are sequential, skip exhaustive pairs
matching_bow_neighbors: 20         # still catch loop closure on revisited areas
feature_process_size_panorama: 4096
```

## Training 360 Gaussian Splatting

First, generate point clouds using images from a 360-degree camera with OpenSfM. Refer to the following repository and use this command for reconstruction:
Visit https://github.com/inuex35/ind-bermuda-opensfm and opensfm documentation for more detail.

```bash
bin/opensfm_run_all your_data
```

Make sure the camera model is set to spherical. It is possible to use both spherical and perspective camera models simultaneously.

`opensfm_run_all` also runs `mesh`, `undistort` and `compute_depthmaps`. Panorama training only needs `reconstruction.json`, so to save considerable time you can stop after `reconstruct`:

```bash
for step in extract_metadata detect_features match_features create_tracks reconstruct; do
    bin/opensfm $step data/your_data
done
```

### Running OpenSfM via Docker

OpenSfM is a separate project with heavy C++ dependencies, so it's easiest to run in Docker. The upstream `Dockerfile` in ind-bermuda-opensfm builds a CUDA image with optional GPU feature matchers (flash-attention, LightGlue, ALIKED, pypopsift) totalling ~15-20GB; note its `TORCH_CUDA_ARCH_LIST` predates Ada (`sm_89`) GPUs. Those matchers are optional — OpenSfM's stock SIFT/HAHOG is sufficient for spherical reconstruction.

`docker/Dockerfile.opensfm-slim` in this repo is a CPU-only alternative (~4GB, builds in a few minutes):

```bash
docker build -t opensfm-slim:latest -f docker/Dockerfile.opensfm-slim docker/
docker run --rm -v $(pwd)/data/your_data:/data opensfm-slim:latest bin/opensfm_run_all /data
```

The dataset directory is bind-mounted at `/data`, and OpenSfM writes its outputs (including `reconstruction.json`) back into it alongside `images/`, which is exactly the layout `train.py` expects.

After reconstruction, a `reconstruction.json` file will be generated. You can use opensfm viewer for visualization.
![image](https://github.com/inuex35/360-gaussian-splatting/assets/129066540/9dbf65e0-3d86-4569-aa82-916cc2ea66d0)


Assuming you are creating directories within `data`, place them as follows:
```
data/your_data/images/*jpg
data/your_data/reconstruction.json
```

Then, start the training with the following command:

```bash
uv run python train.py -s data/your_data --panorama
```

After training, results will be saved in the `output` directory. For training parameters and more details, refer to the Gaussian Splatting repository.

## Training parameter


Parameters for 360 Gaussian Splatting are provided with default values in 360-gaussian-splatting/arguments/__init__.py.

According to the original repository, it might be beneficial to adjust position_lr_init, position_lr_final, and scaling_lr.

Reducing densify_grad_threshold can increase the number of splats, but it will also increase VRAM usage.

densify_from_iter and densify_until_iter are also related to densification.

You should use small densify_grad_threshold like 0.00002 for equirectangular.




