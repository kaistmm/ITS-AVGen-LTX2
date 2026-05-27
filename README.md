## <div align="center">[TMLR] Inference Time Scaling for Joint Audio-Video Generation</div>

<div align="center">

**[Jaemin Jung](https://jung-jaemin.github.io/)**<sup>1</sup>, [Kyeongha Rho](https://kyeongharho.github.io/)<sup>1</sup>, [Inkyu Shin](https://dlsrbgg33.github.io/)<sup>2</sup>, [Joon Son Chung](https://mm.kaist.ac.kr/joon/)<sup>1</sup>

<sup>1</sup> KAIST, <sup>2</sup> Luma AI

[📄 Paper](https://openreview.net/forum?id=MHNFjjm5nO) | [🌐 Project Page](https://openreview.net/forum?id=MHNFjjm5nO) | [📑 Open Review](https://openreview.net/forum?id=MHNFjjm5nO)

</div>

---

<p align="center">
  <img src="assets/src/7_avits.gif" width="70%">
</p>

---

## Brief Introduction

<p align="center">
  <a href="assets/src/main.pdf" target="_blank">
    <img src="assets/src/main.png" width="80%">
  </a>
</p>

**Inference Time Scaling (ITS)** extends generation quality without retraining by leveraging pre-trained reward models at inference time.

Rather than generating a single sample through the diffusion process, ITS generates a population of diverse candidates and uses reward signals (video quality, audio-video synchronization, etc.) to iteratively refine them. This approach enables:

- **BON (Best-of-N)**: Generate N samples and select the best one using reward ranking
- **ARW (Adaptive Reward Weighting)**: Dynamically optimize reward weight combinations for your use case

Key advantages:
- ✅ No additional training required
- ✅ Works with LTX-2 pre-trained models
- ✅ Flexible reward combinations (Video Quality, Temporal Alignment, JavisScore, etc.)
- ✅ Significant quality improvements (up to +34% on sync metrics)

### 🎬 Supported Joint Audio-Video Generation Models

1. **[JavisDiT-ITS](https://github.com/kaistmm/ITS-AVGen)** — Joint Audio-Video Diffusion Transformer
   - Synchronized audio-video generation with spatio-temporal priors

2. **[LTX2-ITS](https://github.com/kaistmm/ITS-AVGen-LTX2)** ⭐ **— Most powerful audio-video generation model**
   - State-of-the-art quality and synchronization
   - Recommended for best results
   - Production-ready outputs with multiple resolution modes
   - **Currently implemented:** BON (Best-of-N)
   - **Coming soon:** EvoSearch (Evolutionary Search)

3. **[MMDisCo-ITS](https://github.com/SonyResearch/MMDisCo)** — Cooperative Diffusion for Joint Audio-Video Generation (TBD)
   - Discriminator-guided multimodal generation
   - *Code will be released soon*

---

> For LTX-2 base model details, installation, and capabilities, refer to the [LTX-2 repository](https://github.com/Lightricks/LTX-2).

---

## Quick Start

### 1. Install LTX-2

Clone this ITS repository:

```bash
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2

# Set up the environment
uv sync --frozen
source .venv/bin/activate
```

### 2. Download Required Models

Download from the [LTX-2.3 HuggingFace repository](https://huggingface.co/Lightricks/LTX-2.3):

**LTX-2.3 Model Checkpoint** (choose one):
```bash
# Development version (better quality)
wget https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors \
    -O ./checkpoints/LTX2.3/ltx-2.3-22b-dev.safetensors

# Or distilled version (faster)
wget https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled.safetensors \
    -O ./checkpoints/LTX2.3/ltx-2.3-22b-distilled.safetensors
```

**Gemma-3 Text Encoder:**

Download from [HuggingFace](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main) and place in `./checkpoints/LTX2/gemma3/`:

**Spatial Upscaler (Optional):**
```bash
wget https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    -O ./checkpoints/LTX2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
```

### 3. Reward Server Environment Setup

The reward server requires additional dependencies for inference time scaling (BON or ARW).

**Separate environment for reward server (recommended):**

```bash
# Create separate conda env
conda create -n reward-server python=3.10
conda activate reward-server

# Install PyTorch (required for this env)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install evaluation dependencies
pip install -r requirements/requirements-eval.txt
pip install imagebind-huge einops ftfy

# Optional: Install CLAP
pip install transformers[audio]>=4.33.0
```

Then run `bash scripts/vqa_server.sh [GPU_ID]` in this separate environment.

### 4. Test Installation

Verify everything works with a quick test:

```bash
bash scripts/inference_one_ltx.sh 0
# This runs standard inference on GPU 0 with sample prompts
```

If successful, output videos will be saved in the configured output directory.

---

## Inference Time Scaling

### Step 1: Start the Reward Server

The reward server must run on a separate GPU:

```bash
bash scripts/vqa_server.sh [GPU_ID]
# Example: bash scripts/vqa_server.sh 0
```

This launches reward models on port 5002. Keep this running during generation.

**Server Configuration Options:**

Edit `scripts/vqa_server.sh` or pass arguments directly:

```bash
CUDA_VISIBLE_DEVICES=0 python reward_model/vqa_server.py \
    --gpu 0 \
    --addr 5002 \
    --reward_model [REWARD_MODEL] \
    --align_model [ALIGN_MODEL]
```

| Option | Values | Default |
|--------|--------|---------|
| `--reward_model` | `VideoReward`, `vqascore` | `VideoReward` |
| `--align_model` | `JavisScore`, `AVHScore`, `AVIB`, `All`, `None` | `JavisScore` |

**Examples:**
```bash
# Default (VideoReward + JavisScore)
bash scripts/vqa_server.sh 0

# With all alignment models
CUDA_VISIBLE_DEVICES=0 python reward_model/vqa_server.py \
    --gpu 0 --addr 5002 \
    --reward_model VideoReward \
    --align_model All
```

### Step 2: Run Inference

#### Option A: Standard (No ITS)

Generate without inference time scaling:

```bash
bash scripts/inference_one_ltx.sh [GPU_ID] [NSHARD] [SHARD_ID]
# Single GPU: bash scripts/inference_one_ltx.sh 0
# Multi-GPU: for i in {0..3}; do bash scripts/inference_one_ltx.sh $i 4 $i & done; wait
```

**Features:**
- No ITS - pure LTX-2 generation
- Single sample per prompt
- Fast inference

#### Option B: BON (Best-of-N)

Generate N candidates, select best:

```bash
bash scripts/inference_one_ltx_bon.sh [GPU_ID] [NSHARD] [SHARD_ID]
# Single GPU: bash scripts/inference_one_ltx_bon.sh 0
# Multi-GPU: for i in {0..3}; do bash scripts/inference_one_ltx_bon.sh $i 4 $i & done; wait
```

**Config** (`scripts/inference_one_ltx_bon_config.json`):
- `BON_SAMPLES` → Number of candidates (default: 10)
- `BON_AGGREGATION_METHOD` → "arw" (Adaptive Reward Weighting)
- `BON_REWARD_KEY` → Primary reward metric (TA, VR, etc.)
- `BON_ALIGN_KEY` → Alignment metric (JS, etc.)

---

## Configuration

BON settings are configured in `scripts/inference_one_ltx_bon_config.json`:

**Reward Models (Metrics):**
```json
{
  "BON_REWARD_KEY": "TA",
  "BON_ALIGN_KEY": "JS",
  "BON_REWARD_WEIGHT": 0.5,
  "BON_ALIGN_WEIGHT": 0.5
}
```

Available reward metrics:
| Metric | Purpose | Description |
|--------|---------|-------------|
| `TA` | Temporal Alignment | Audio-video sync quality |
| `JS` | JavisScore | Audio-visual harmony |
| `VR` | VideoReward | Overall video quality |

Example with multiple metrics:
```json
{
  "BON_SAMPLES": 10,
  "BON_REWARD_KEY": "TA",
  "BON_ALIGN_KEY": "JS",
  "BON_REWARD_WEIGHT": 0.4,
  "BON_ALIGN_WEIGHT": 0.6,
  "BON_AGGREGATION_METHOD": "arw"
}
```

**Aggregation Methods:**

BON supports multiple aggregation strategies to combine reward scores:

| Method | Description |
|--------|-------------|
| `weighted` | Weighted linear combination of metrics (default) |
| `rank` | Rank-based normalization before aggregation |
| `arw` | Adaptive Reward Weighting with learnable weights (recommended) |

**ARW (Adaptive Reward Weighting):**

Dynamically optimize reward weights during inference for best performance:

```json
{
  "BON_AGGREGATION_METHOD": "arw",
  "BON_ARW_LR": 0.05,
  "BON_ARW_MAX_ITER": 5,
  "BON_ARW_OPTIMIZER": "Adam"
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BON_ARW_LR` | 0.05 | Learning rate for weight optimization (0.01-0.1) |
| `BON_ARW_MAX_ITER` | 5 | Number of optimization iterations (1-10) |
| `BON_ARW_OPTIMIZER` | Adam | Optimizer type (Adam, SGD) |

---

## Performance Optimization

### FP8 Quantization

Enable lower memory footprint:

```bash
USE_FP8_CAST=1 bash scripts/inference_one_ltx_bon.sh 0
```

### Model Reuse

Optimize for multiple prompt processing:

```bash
BATCH_MODEL_MODE=reuse bash scripts/inference_one_ltx_bon.sh 0
```

Options:
- `reuse`: Load model once, process all prompts (memory efficient)
- `reload`: Load/unload for each prompt (slower, less VRAM)

### Multi-GPU Distributed Processing

Process large datasets efficiently:

```bash
# Process dataset in 4 shards
for i in {0..3}; do
    bash scripts/inference_one_ltx_bon.sh $i 4 $i &
done
wait
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused (5002) | Ensure `vqa_server.sh` is running on separate GPU |
| Out of memory | Reduce `BON_SAMPLES` in config, enable FP8 quantization, or use reuse mode |
| CUDA errors | Check CUDA 12.1 compatibility and sufficient VRAM (min. 48GB for dev model) |
| Slow generation | Enable FP8 quantization, model reuse mode, or reduce BON_SAMPLES |

---

## Citation

```bibtex
@article{its2024,
    title={Inference-Time Scaling for Joint Audio--Video Generation},
    author={Jung, Jaemin and Rho, Kyeongha and Shin, Inkyu and Chung, Joon Son},
    journal={Transactions on Machine Learning Research},
    year={2024}
}
```

---

## Reference

- **LTX-2 Repository**: https://github.com/Lightricks/LTX-2
- **LTX-2 Model Card**: https://huggingface.co/Lightricks/LTX-2.3
- **LTX-2 Paper**: https://arxiv.org/abs/2601.03233
- **JavisDiT**: https://github.com/JavisVerse/JavisDiT
- **Paper**: https://openreview.net/forum?id=MHNFjjm5nO

For questions, open an issue or contact the authors.
