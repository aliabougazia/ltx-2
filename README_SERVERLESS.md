# LTX-2 Serverless Training on RunPod

Deploy LTX-2 LoRA training as a serverless endpoint on RunPod.

## Overview

This setup allows you to:
- Train LTX-2 LoRA models via API calls
- Scale automatically based on demand
- Pay only for actual training time
- Deploy once, use many times

## Setup

### 1. Build and Push Docker Image

```bash
cd /home/aa/LTX-2

# Build with GitHub token
docker build \
  --build-arg GITHUB_TOKEN=your_github_token_here \
  -t your-dockerhub-username/ltx-2-serverless:latest \
  -f Dockerfile .

# Push to Docker Hub
docker login
docker push your-dockerhub-username/ltx-2-serverless:latest
```

### 2. Deploy on RunPod Serverless

1. Go to: https://runpod.io/console/serverless
2. Click **"New Endpoint"**
3. Configure:
   - **Name**: ltx-2-trainer
   - **Docker Image**: `your-dockerhub-username/ltx-2-serverless:latest`
   - **Container Disk**: 20GB
   - **GPU Types**: RTX 4090, RTX 5090, A100
   - **Min Workers**: 0 (auto-scale)
   - **Max Workers**: 3 (adjust based on budget)
   - **Idle Timeout**: 30 seconds
   - **Execution Timeout**: 3600 seconds (1 hour) or more

4. **Network Volume** (Optional but Recommended):
   - Attach a network volume with your model checkpoints
   - Mount at `/workspace/checkpoints`
   - Pre-upload: `ltx-2-19b-dev.safetensors` and `gemma-3-12b-it-qat-q4_0-unquantized/`

5. **Environment Variables** (Optional):
   ```
   WANDB_API_KEY=your_wandb_key
   HF_TOKEN=your_huggingface_token
   ```

6. Click **Deploy**

### 3. Upload Training Data

Upload your preprocessed dataset to the network volume:
```
/workspace/datasets/your-dataset/.precomputed/
├── latents/
├── conditions/
└── audio_latents/
```

Or use RunPod's file upload API before training.

## Usage

### Python Client

```python
import runpod

# Set your API key
runpod.api_key = "your_runpod_api_key"

# Get your endpoint
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Start training
job = endpoint.run({
    "input": {
        "task": "train",
        
        # Paths (use network volume paths)
        "model_path": "/workspace/checkpoints/ltx-2-19b-dev.safetensors",
        "text_encoder_path": "/workspace/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized",
        "preprocessed_data_root": "/workspace/datasets/your-dataset/.precomputed/",
        "output_dir": "outputs/my-lora-training",
        
        # LoRA configuration
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        
        # Training configuration
        "steps": 3000,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "with_audio": True,
        
        # Hardware configuration
        "quantization": "int8-quanto",
        "mixed_precision_mode": "bf16",
        "enable_gradient_checkpointing": True,
        
        # Validation (optional, set to null to disable)
        "validation_prompts": [
            "A cat playing with a ball",
            "A dog running in a field"
        ],
        "validation_interval": None,  # null = disabled during training
        
        # Checkpointing
        "checkpoint_interval": 250,
        "keep_last_n": 3,
        
        # Weights & Biases (optional)
        "wandb_enabled": False,
        "wandb_project": "ltx-2-training",
        
        # Hugging Face Hub (optional)
        "push_to_hub": False,
        "hub_model_id": "your-username/my-lora-model",
    }
})

# Check status
print(f"Job ID: {job.job_id}")
print(f"Status: {job.status()}")

# Wait for completion
result = job.output()
print(result)

# Download trained checkpoint
if result["status"] == "success":
    checkpoint_path = result["checkpoint_path"]
    print(f"Training complete! Checkpoint: {checkpoint_path}")
    print(f"Checkpoint size: {result['checkpoint_size_mb']:.2f} MB")
```

### cURL Example

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "train",
      "model_path": "/workspace/checkpoints/ltx-2-19b-dev.safetensors",
      "text_encoder_path": "/workspace/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized",
      "preprocessed_data_root": "/workspace/datasets/my-data/.precomputed/",
      "steps": 3000,
      "batch_size": 1,
      "gradient_accumulation_steps": 4,
      "learning_rate": 0.0001,
      "quantization": "int8-quanto",
      "validation_interval": null
    }
  }'
```

## Job Input Parameters

### Required
- `task`: Task type (`"train"`, `"validate"`, `"inference"`)
- `model_path`: Path to LTX-2 checkpoint
- `text_encoder_path`: Path to Gemma text encoder
- `preprocessed_data_root`: Path to preprocessed training data

### Optional Training Config
- `steps`: Training steps (default: 3000)
- `batch_size`: Batch size per GPU (default: 1)
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `learning_rate`: Learning rate (default: 1e-4)
- `with_audio`: Train with audio (default: true)

### Optional LoRA Config
- `lora_rank`: LoRA rank (default: 32)
- `lora_alpha`: LoRA alpha (default: 32)
- `lora_dropout`: LoRA dropout (default: 0.0)
- `lora_target_modules`: Modules to apply LoRA (default: attention layers)

### Optional Hardware Config
- `quantization`: Quantization method (default: "int8-quanto")
- `mixed_precision_mode`: Precision mode (default: "bf16")
- `enable_gradient_checkpointing`: Enable gradient checkpointing (default: true)

### Optional Validation Config
- `validation_prompts`: List of prompts (default: [])
- `validation_interval`: Steps between validation (default: null = disabled)
- `inference_steps`: Inference steps for validation (default: 30)

### Optional Checkpoint Config
- `checkpoint_interval`: Steps between checkpoints (default: 250)
- `keep_last_n`: Number of checkpoints to keep (default: -1 = all)

### Optional Integrations
- `wandb_enabled`: Enable W&B logging (default: false)
- `wandb_api_key`: W&B API key (or set as env var)
- `push_to_hub`: Push to HF Hub (default: false)
- `hub_model_id`: HF Hub model ID
- `hf_token`: HF token (or set as env var)

## Cost Estimation

### Per Training Session (3000 steps)

| GPU | Training Time | Cost/hr | Total Cost |
|-----|---------------|---------|------------|
| RTX 4090 | ~5 hours | $0.34 | ~$1.70 |
| RTX 5090 | ~4 hours | $0.50 | ~$2.00 |
| A100 (40GB) | ~3 hours | $1.00 | ~$3.00 |

**Serverless Benefits:**
- No idle costs (workers shut down automatically)
- Scale to 0 when not in use
- Pay only for actual compute time
- Network volume storage: ~$0.10/GB/month

## Monitoring

### Check Job Status

```python
status = job.status()
print(status)
```

### Stream Logs

```python
# Get logs
logs = job.stream()
for log in logs:
    print(log)
```

### List Jobs

```python
jobs = endpoint.list_jobs()
for j in jobs:
    print(f"{j.job_id}: {j.status()}")
```

## Downloading Results

### Option 1: Via RunPod API

```python
# After training completes
result = job.output()
checkpoint_path = result["checkpoint_path"]

# Download from network volume
# (Implement download logic based on your storage setup)
```

### Option 2: Push to Hugging Face Hub

Set in job input:
```python
"push_to_hub": True,
"hub_model_id": "your-username/my-lora-model",
"hf_token": "your_hf_token"
```

Then download from HF:
```bash
huggingface-cli download your-username/my-lora-model
```

## Troubleshooting

### Worker Not Starting
- Check Docker image is public or credentials are set
- Verify network volume is attached correctly
- Check execution timeout is sufficient (3600s+)

### Out of Memory
Reduce in job input:
```python
"batch_size": 1,
"gradient_accumulation_steps": 8,
"with_audio": False,
"quantization": "int8-quanto"
```

### Slow Training
- Use faster GPUs (A100 > RTX 5090 > RTX 4090)
- Increase worker max workers for parallel training
- Pre-upload data to network volume

### Checkpoint Not Found
- Check `output_dir` path is writable
- Verify checkpoint_interval is set
- Ensure training completed successfully

## Best Practices

1. **Use Network Volumes**: Pre-upload models and datasets
2. **Disable Validation**: Set `validation_interval: null` for faster training
3. **Enable Checkpointing**: Save progress every 250 steps
4. **Set Timeouts**: Allow enough time for training to complete
5. **Monitor Costs**: Use min_workers: 0 to avoid idle charges
6. **Test First**: Run short training (100 steps) to verify setup

## Support

- RunPod Docs: https://docs.runpod.io/serverless
- Discord: https://discord.gg/runpod
- GitHub Issues: https://github.com/aliabougazia/ltx-2/issues
