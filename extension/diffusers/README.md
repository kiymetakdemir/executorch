# Diffusers

This directory provides experimental export and Python runtime support for
Hugging Face checkpoints using `StableDiffusionPipeline`,
`StableDiffusionXLPipeline`, or `FluxPipeline`. The exported PTE stores
the text encoder, denoiser and VAE methods together with the metadata needed
by the runner.

All Python APIs in this package are experimental and may change without notice.

## Installation

Install the optional diffusion dependencies:

```bash
python -m pip install -r extension/diffusers/requirements.txt
```

## Usage

### Export

Run from the ExecuTorch repository root:

```bash
python -m executorch.export.huggingface.diffusers.export \
  stable-diffusion-v1-5/stable-diffusion-v1-5 \
  stable_diffusion_mlx.pte \
  --backend mlx \
  --denoiser-dtype float16
```

Use `stabilityai/stable-diffusion-xl-base-1.0` to export SDXL.

FLUX.1-schnell and FLUX.1-dev are discovered from `model_index.json` and use
their native defaults: four steps without embedded guidance for Schnell, and
28 steps with guidance scale 3.5 for Dev. FLUX supports static or bounded
dynamic text-to-image export. Heights, widths, and dynamic size bounds must be
multiples of twice the VAE scale factor (normally 16), and
`--max-sequence-length` changes only the static T5 sequence length. Both static
and dynamic FLUX exports precompute reference RoPE axis tables and select the
requested image coordinates at runtime. Static tables cover the exported image
dimensions; dynamic tables cover `--max-size`. True classifier-free
guidance is enabled by passing both
`--cfg-scale` greater than one and an explicit `--negative-prompt`; the
independent `--guidance-scale` option controls Dev's embedded guidance input.

Portable FLUX export and execution are supported. A synthetic FLUX transformer
also lowers completely to MLX, but full-checkpoint MLX coverage depends on the
operators used by the selected transformer and is not guaranteed yet.

### CUDA

On a CUDA-enabled machine with the ExecuTorch CUDA backend installed, export
FLUX.1-dev with:

```bash
python -m executorch.export.huggingface.diffusers.export \
  black-forest-labs/FLUX.1-dev flux_dev_cuda.pte \
  --backend cuda \
  --height 1024 --width 1024 \
  --max-sequence-length 512 \
  --text-encoder-dtype float32 \
  --denoiser-dtype bfloat16 \
  --vae-dtype float32
```

Add `--dynamic-shapes --min-size 512 --max-size 1024` for bounded dynamic
image dimensions. CUDA writes external data alongside the PTE; pass the
generated `.ptd` file to the runner using `--data-path`.

CUDA also supports denoiser quantization through `--qlinear` and
`--qlinear-group-size`. Use `--qlinear 4w` with `--denoiser-dtype bfloat16`
for the packed INT4 AOTI path. Text encoder quantization remains deferred.

### Run

```bash
python -m executorch.extension.diffusers.runner \
  stable_diffusion_mlx.pte \
  --model-id stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --prompt "a photo of a cat" \
  --steps 50 \
  --seed 9 \
  --output cat.png
```

The runner downloads the matching Hugging Face tokenizers and scheduler
specified by `--model-id`.

## Configuration options

### Dynamic shapes

Export SDXL with dynamic spatial dimensions:

```bash
python -m executorch.export.huggingface.diffusers.export \
  stabilityai/stable-diffusion-xl-base-1.0 \
  sdxl_dynamic_mlx.pte \
  --backend mlx \
  --denoiser-dtype float16 \
  --vae-dtype float32 \
  --dynamic-shapes \
  --min-size 512 \
  --max-size 1024
```

Run it at a resolution within that range:

```bash
python -m executorch.extension.diffusers.runner \
  sdxl_dynamic_mlx.pte \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "an astronaut riding a horse" \
  --height 768 \
  --width 1024 \
  --output astronaut.png
```

### Quantization

Quantize the denoiser's linear weights during MLX export:

```bash
python -m executorch.export.huggingface.diffusers.export \
  stabilityai/stable-diffusion-xl-base-1.0 \
  sdxl_4w_mlx.pte \
  --backend mlx \
  --denoiser-dtype float16 \
  --vae-dtype float32 \
  --qlinear 4w \
  --qlinear-group-size 64
```

MLX linear quantization supports `4w`, `8w` and `nvfp4`; supported group sizes
are 32, 64 and 128.
