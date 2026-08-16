# Benchmarks

Timing records for the 360 capture → Gaussian Splat pipeline, kept so CPU and GPU
runs can be compared on identical input.

## Machine

| | |
| --- | --- |
| CPU | AMD Ryzen 7 3800X, 8 cores / 16 threads, 4.56 GHz max |
| RAM | 15 GB total (~6 GB available during these runs) |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8 GB VRAM, driver 560.35.05 |
| CUDA | toolkit 12.6, torch 2.13.0+cu126 |
| Python | 3.14 (uv) |
| Dataset volume | `/dev/nvme0n1p2`, NTFS (ntfs3) |
| OS | Linux 6.8.0 |

## Dataset: `R0010043`

| | |
| --- | --- |
| Source | `R0010043.MP4`, RICOH THETA X |
| Video | 3840×1920 equirectangular, 44.19 s, 30 fps, h264, 300 MB |
| Extraction | `ffmpeg -vf fps=3 -q:v 2` → **133 frames**, 3840×1920 JPEG, 86 MB |
| Features | HAHOG, mean **19,625** points/image |
| Camera model | spherical (forced via `camera_models_overrides.json`; frames have no EXIF) |

OpenSfM config: `processes: 16`, `matching_order_neighbors: 20`,
`matching_bow_neighbors: 20`, `feature_process_size_panorama: 4096`.

## Run 1 — OpenSfM reconstruction, CPU only

Image: `opensfm-slim:latest` (3.98 GB), CPU-only build of ind-bermuda-opensfm
(`docker/Dockerfile.opensfm-slim`). No GPU feature matchers.

Docker image build: **411 s** cold, **128 s** warm (cached layers).

Per-step wall time, from OpenSfM's own `profile.log`:

| Step | Time | Notes |
| --- | --- | --- |
| `extract_metadata` | 1.08 s | |
| `detect_features` | 89.6 s | 133 images, HAHOG @ 4096 px |
| `match_features` | 374.4 s | 2,309 pairs, mean 173 robust matches/pair |
| `create_tracks` | 5.14 s | |
| `reconstruct` | 210.4 s | incremental SfM + bundle adjustment |
| **Total** | **680.7 s** (11 min 21 s) | |

Result: a single reconstruction containing **all 133 images** and **38,997 points** —
no fragmentation and no dropped frames, so the capture registered cleanly.

`match_features` is the dominant CPU stage at 6.2 min — 2,309 pairs rather than the
8,778 an exhaustive run would need, thanks to `matching_order_neighbors` +
`matching_bow_neighbors`. This is the stage GPU matchers would target.

Skipped from `opensfm_run_all`: `mesh`, `undistort`, `compute_depthmaps` — panorama
training only consumes `reconstruction.json`, and depthmaps dominate runtime.

## Run 2 — Gaussian Splatting training (baseline, default params)

| | |
| --- | --- |
| Command | `python train.py -s data/R0010043 --panorama -m output/R0010043_bench` |
| Iterations | 30,000 |
| Input resolution | 1600×800 (auto-downscaled from 3840×1920 by `resolution=-1`) |
| **Wall time** | **25 min 08 s** (1507.6 s) |
| Camera load | ~25 s of that (133 images) |
| **Peak VRAM** | **3943 MiB / 8188 MiB** |
| Peak host RSS | 4.87 GB |
| Initial points | 38,997 (from SfM) |
| **Final splats** | **119,498** (3.1× growth), 29 MB ply |
| Exit | 0 |

Throughput between checkpoints, after densification stops at iteration 15,000:

| Interval | Time | Rate |
| --- | --- | --- |
| 10k → 20k | 512 s | 19.5 it/s |
| 20k → 30k | 517 s | 19.3 it/s |

### Caveat on this baseline

Run at the stock `densify_grad_threshold = 0.0002`. The README recommends
**0.00002** for equirectangular data, and the 3.1× densification growth here is
low, which is the symptom that threshold is meant to fix. Peak VRAM was only
3.9 GB of 8 GB, so there is substantial headroom to rerun at the lower threshold —
expect more splats, more VRAM, and a longer run. Treat this run as a *timing*
baseline rather than a quality target.

Full-resolution (3840 px) training is not feasible on this machine: 133 native
frames need ~11.7 GB on GPU, and `--data_device cpu` cannot absorb it with ~6 GB
of host RAM free.

### Output integrity

21 of 119,498 splats (0.018%) finished with non-finite `x/y/z`. The fraction is
negligible and the remaining cloud is well-formed (extent 248×307×225, centroid
near origin), but some viewers choke on NaN positions, so filter them before
export if a viewer misbehaves. Mean opacity 0.244, with 22,839 splats above 0.5.

## End-to-end

| Phase | Time |
| --- | --- |
| Frame extraction (ffmpeg, 3 fps) | ~9 s |
| OpenSfM reconstruction (CPU) | 680.7 s |
| Gaussian Splatting training (GPU) | 1507.6 s |
| **Total** | **~36.5 min** |

## Notes for a GPU comparison

The upstream ind-bermuda-opensfm `Dockerfile` adds GPU feature matchers
(pypopsift SIFT, LightGlue, ALIKED, flash-attention). To benchmark those against
this CPU baseline, two things need changing first:

- `TORCH_CUDA_ARCH_LIST` on line 30 lists only up to Turing; Ada (`sm_89`, the
  4060 Ti) must be added or the kernels won't target this GPU.
- The image is ~15–20 GB and Docker's root is on `/`, which had ~5.5 GB free after
  the slim build. Free space or relocate `data-root` before attempting it.

Feature detection and matching are the stages GPU acceleration targets;
`reconstruct` is largely serial bundle adjustment and will not speed up much.
