# LTX-2 Training on RunPod

This guide shows you how to train LTX-2 LoRA models on RunPod using Docker.

## Prerequisites

1. **GitHub Personal Access Token** with repo access
   - Go to: https://github.com/settings/tokens
   - Generate a new token (classic) with `repo` scope
   - Save the token securely

2. **RunPod Account** with credits
   - Sign up at: https://runpod.io

## Option 1: Build and Push Docker Image

### Step 1: Build the Docker Image Locally

```bash
cd /home/aa/LTX-2

# Build the image (replace YOUR_GITHUB_TOKEN)
docker build \
  --build-arg GITHUB_TOKEN=your_github_token_here \
  -t your-dockerhub-username/ltx-2-trainer:latest \
  -f Dockerfile .
```

### Step 2: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push the image
docker push your-dockerhub-username/ltx-2-trainer:latest
```

### Step 3: Use on RunPod

1. Go to RunPod → Pods → Deploy
2. Select GPU (RTX 4090, A100, or H100 recommended)
3. Use Custom Docker Image: `your-dockerhub-username/ltx-2-trainer:latest`
4. Set Volume Size: 50GB+ (for checkpoints and datasets)
5. Deploy

## Option 2: Build Directly on RunPod

### Step 1: Deploy a Pod with NVIDIA Base Image

1. Go to RunPod → Pods → Deploy
2. Select GPU: RTX 4090/5090, A100, or H100
3. Use Image: `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
4. Volume: 50GB+
5. Deploy and Connect via SSH or Jupyter

### Step 2: Setup Inside the Pod

```bash
# Install system dependencies
apt-get update && apt-get install -y \
    git curl build-essential python3.11 python3.11-dev \
    python3.11-venv ffmpeg libsndfile1 ninja-build

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.cargo/bin:${PATH}"

# Clone your repo (replace with your token)
git clone https://YOUR_GITHUB_TOKEN@github.com/aliabougazia/ltx-2.git /workspace/ltx-2
cd /workspace/ltx-2

# Install dependencies
uv sync
```

## Training Setup

### 1. Upload Model Checkpoints

Upload to `/workspace/checkpoints/`:
- `ltx-2-19b-dev.safetensors`
- `gemma-3-12b-it-qat-q4_0-unquantized/` (directory)

### 2. Prepare Your Dataset

Upload preprocessed data to `/workspace/datasets/`:
```
/workspace/datasets/your-dataset/
├── latents/
├── conditions/
└── audio_latents/  (if training with audio)
```

### 3. Create Training Config

```bash
cd /workspace/ltx-2/packages/ltx-trainer

# Copy and edit config
cp configs/ltx2_av_lora.yaml datasets/your-dataset/config.yaml

# Edit paths in config.yaml:
# - model_path: "/workspace/checkpoints/ltx-2-19b-dev.safetensors"
# - text_encoder_path: "/workspace/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized"
# - preprocessed_data_root: "/workspace/datasets/your-dataset/.precomputed/"
# - output_dir: "outputs/your-training-run"
```

### 4. Start Training

```bash
cd /workspace/ltx-2/packages/ltx-trainer

# Activate environment
source /workspace/ltx-2/.venv/bin/activate

# Start training
uv run python scripts/train.py datasets/your-dataset/config.yaml
```

## Recommended RunPod Settings

### GPU Options (by budget)

| GPU | VRAM | Cost/hr | Batch Size | Notes |
|-----|------|---------|------------|-------|
| RTX 4090 | 24GB | ~$0.34 | 1 | Minimum for training |
| RTX 5090 | 32GB | ~$0.50 | 1-2 | Good balance |
| A100 (40GB) | 40GB | ~$1.00 | 1-2 | Fastest training |
| A100 (80GB) | 80GB | ~$1.50 | 2-4 | Best for large batches |

### Volume Settings

- **Minimum**: 50GB (for model + small dataset)
- **Recommended**: 100GB+ (for larger datasets + outputs)
- **Network Volume**: Enable for persistent storage across pods

### Environment Variables (Optional)

Set in Pod configuration:
```bash
WANDB_API_KEY=your_wandb_key  # For experiment tracking
HF_TOKEN=your_hf_token        # For Hugging Face Hub uploads
```

## Monitoring Training

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View Logs
```bash
cd /workspace/ltx-2/packages/ltx-trainer
tail -f outputs/your-training-run/*.log
```

### Access Checkpoints
Checkpoints are saved to:
```
/workspace/ltx-2/packages/ltx-trainer/outputs/your-training-run/checkpoints/
```

Download them via RunPod's file manager or rsync.

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `batch_size` in config (try 1)
2. Increase `gradient_accumulation_steps` (try 8-16)
3. Disable audio training: `with_audio: false`
4. Use smaller video dimensions in dataset
5. Set `quantization: "int8-quanto"` (with our fix)

### Slow Training
1. Check GPU utilization: `nvidia-smi`
2. Increase `num_dataloader_workers` (try 4)
3. Enable gradient checkpointing: `enable_gradient_checkpointing: true`
4. Use persistent network volumes (faster I/O)

### Connection Drops
1. Use `tmux` or `screen` to keep training running
2. Enable checkpoint saving: `checkpoints.interval: 250`
3. Use RunPod's auto-save feature

## Cost Estimation

Example 3000-step training:
- **RTX 4090**: ~6 hours × $0.34/hr = **~$2.04**
- **RTX 5090**: ~5 hours × $0.50/hr = **~$2.50**
- **A100 (40GB)**: ~3 hours × $1.00/hr = **~$3.00**

Add ~$0.50-1.00 for storage (100GB for 1 week)

## Support

For issues, check:
- LTX-2 Trainer Docs: `/workspace/ltx-2/packages/ltx-trainer/docs/`
- RunPod Discord: https://discord.gg/runpod
- GitHub Issues: https://github.com/aliabougazia/ltx-2/issues
