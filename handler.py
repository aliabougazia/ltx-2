"""
RunPod Serverless Handler for LTX-2 LoRA Training

This handler allows you to train LTX-2 LoRA models via RunPod's serverless API.
"""

import runpod
import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add packages to path
sys.path.insert(0, "/workspace/ltx-2/packages/ltx-trainer/src")
sys.path.insert(0, "/workspace/ltx-2/packages/ltx-core/src")
sys.path.insert(0, "/workspace/ltx-2/packages/ltx-pipelines/src")


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def prepare_config(job_input: Dict[str, Any]) -> Path:
    """Prepare training configuration from job input."""
    
    # Default config paths
    model_path = job_input.get("model_path", "/workspace/checkpoints/ltx-2-19b-dev.safetensors")
    text_encoder_path = job_input.get("text_encoder_path", "/workspace/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized")
    data_path = job_input.get("preprocessed_data_root", "/workspace/datasets/.precomputed/")
    output_dir = job_input.get("output_dir", "outputs/serverless-training")
    
    # Build config dict
    config = {
        "model": {
            "model_path": model_path,
            "text_encoder_path": text_encoder_path,
            "training_mode": job_input.get("training_mode", "lora"),
            "load_checkpoint": job_input.get("load_checkpoint", None),
        },
        "lora": {
            "rank": job_input.get("lora_rank", 32),
            "alpha": job_input.get("lora_alpha", 32),
            "dropout": job_input.get("lora_dropout", 0.0),
            "target_modules": job_input.get("lora_target_modules", [
                "to_k", "to_q", "to_v", "to_out.0"
            ]),
        },
        "training_strategy": {
            "name": job_input.get("training_strategy", "text_to_video"),
            "first_frame_conditioning_p": job_input.get("first_frame_conditioning_p", 0.5),
            "with_audio": job_input.get("with_audio", True),
        },
        "optimization": {
            "learning_rate": job_input.get("learning_rate", 1e-4),
            "steps": job_input.get("steps", 3000),
            "batch_size": job_input.get("batch_size", 1),
            "gradient_accumulation_steps": job_input.get("gradient_accumulation_steps", 4),
            "max_grad_norm": job_input.get("max_grad_norm", 1.0),
            "optimizer_type": job_input.get("optimizer_type", "adamw8bit"),
            "scheduler_type": job_input.get("scheduler_type", "cosine"),
            "scheduler_params": job_input.get("scheduler_params", {}),
            "enable_gradient_checkpointing": job_input.get("enable_gradient_checkpointing", True),
        },
        "acceleration": {
            "mixed_precision_mode": job_input.get("mixed_precision_mode", "bf16"),
            "quantization": job_input.get("quantization", "int8-quanto"),
            "load_text_encoder_in_8bit": job_input.get("load_text_encoder_in_8bit", True),
        },
        "data": {
            "preprocessed_data_root": data_path,
            "num_dataloader_workers": job_input.get("num_dataloader_workers", 2),
        },
        "validation": {
            "prompts": job_input.get("validation_prompts", []),
            "negative_prompt": job_input.get("negative_prompt", "worst quality, inconsistent motion, blurry"),
            "images": None,
            "reference_videos": None,
            "video_dims": job_input.get("video_dims", [576, 576, 89]),
            "frame_rate": job_input.get("frame_rate", 25.0),
            "seed": job_input.get("seed", 42),
            "inference_steps": job_input.get("inference_steps", 30),
            "interval": job_input.get("validation_interval", None),  # null = disabled
            "videos_per_prompt": 1,
            "guidance_scale": job_input.get("guidance_scale", 4.0),
            "stg_scale": job_input.get("stg_scale", 1.0),
            "stg_blocks": job_input.get("stg_blocks", [29]),
            "stg_mode": job_input.get("stg_mode", "stg_av"),
            "generate_audio": job_input.get("generate_audio", True),
            "skip_initial_validation": True,
        },
        "checkpoints": {
            "interval": job_input.get("checkpoint_interval", 250),
            "keep_last_n": job_input.get("keep_last_n", -1),
        },
        "flow_matching": {
            "timestep_sampling_mode": job_input.get("timestep_sampling_mode", "shifted_logit_normal"),
            "timestep_sampling_params": {},
        },
        "hub": {
            "push_to_hub": job_input.get("push_to_hub", False),
            "hub_model_id": job_input.get("hub_model_id", None),
        },
        "wandb": {
            "enabled": job_input.get("wandb_enabled", False),
            "project": job_input.get("wandb_project", "ltx-2-trainer"),
            "entity": job_input.get("wandb_entity", None),
            "tags": job_input.get("wandb_tags", ["ltx2", "lora", "serverless"]),
            "log_validation_videos": False,
        },
        "seed": job_input.get("seed", 42),
        "output_dir": output_dir,
    }
    
    # Save config to file
    config_path = Path("/tmp/serverless_training_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def train_model(config_path: Path) -> Dict[str, Any]:
    """Execute training with the given config."""
    
    # Change to trainer directory
    trainer_dir = Path("/workspace/ltx-2/packages/ltx-trainer")
    os.chdir(trainer_dir)
    
    # Run training
    cmd = [
        sys.executable,
        "scripts/train.py",
        str(config_path),
    ]
    
    result = {
        "status": "starting",
        "output": [],
        "error": None,
    }
    
    try:
        # Run training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Capture output
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.strip())
            # Keep last 100 lines to avoid memory issues
            if len(output_lines) > 100:
                output_lines.pop(0)
        
        process.wait()
        
        result["output"] = output_lines
        
        if process.returncode == 0:
            result["status"] = "success"
            
            # Find the saved checkpoint
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            output_dir = Path(config["output_dir"])
            checkpoints = list(output_dir.glob("checkpoints/*.safetensors"))
            
            if checkpoints:
                # Sort by modification time, get latest
                latest_checkpoint = sorted(checkpoints, key=lambda p: p.stat().st_mtime)[-1]
                result["checkpoint_path"] = str(latest_checkpoint)
                result["checkpoint_size_mb"] = latest_checkpoint.stat().st_size / (1024 * 1024)
            
        else:
            result["status"] = "failed"
            result["error"] = f"Training process exited with code {process.returncode}"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def handler(job):
    """
    RunPod serverless handler function.
    
    Job Input Schema:
    {
        "task": "train",  # or "validate", "inference"
        
        # Training parameters
        "model_path": "/workspace/checkpoints/ltx-2-19b-dev.safetensors",
        "text_encoder_path": "/workspace/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized",
        "preprocessed_data_root": "/workspace/datasets/.precomputed/",
        "output_dir": "outputs/my-training",
        
        # LoRA config
        "lora_rank": 32,
        "lora_alpha": 32,
        "training_mode": "lora",
        
        # Training config
        "steps": 3000,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "with_audio": true,
        
        # Hardware config
        "quantization": "int8-quanto",
        "mixed_precision_mode": "bf16",
        
        # Optional: validation
        "validation_prompts": ["a cat playing"],
        "validation_interval": null,  # null = disabled
        
        # Optional: W&B logging
        "wandb_enabled": false,
        "wandb_api_key": "xxx",  # Set as environment variable
        
        # Optional: HF Hub
        "push_to_hub": false,
        "hub_model_id": "username/model-name",
        "hf_token": "xxx"  # Set as environment variable
    }
    """
    
    job_input = job["input"]
    task = job_input.get("task", "train")
    
    try:
        # Set environment variables if provided
        if "wandb_api_key" in job_input:
            os.environ["WANDB_API_KEY"] = job_input["wandb_api_key"]
        if "hf_token" in job_input:
            os.environ["HF_TOKEN"] = job_input["hf_token"]
        
        if task == "train":
            # Prepare config
            config_path = prepare_config(job_input)
            
            # Train model
            result = train_model(config_path)
            
            return result
        
        elif task == "validate":
            return {
                "status": "error",
                "error": "Validation task not yet implemented"
            }
        
        elif task == "inference":
            return {
                "status": "error", 
                "error": "Inference task not yet implemented"
            }
        
        else:
            return {
                "status": "error",
                "error": f"Unknown task: {task}"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": str(e.__traceback__),
        }


if __name__ == "__main__":
    # Start the serverless worker
    runpod.serverless.start({"handler": handler})
