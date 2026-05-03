"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import sys
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")
    # Hope/NL Stage 1+: pre-Stage-1 checkpoints have no memory layer.
    if "hope_memory_layer" not in model_config_kwargs:
        model_config_kwargs["hope_memory_layer"] = None
    # Hope/NL Stage 1-additive: pre-Stage-1.5 checkpoints have no additive memory layer.
    if "hope_additive_memory_layer" not in model_config_kwargs:
        model_config_kwargs["hope_additive_memory_layer"] = None
    # Hope/NL Stage 1.5b: W_o init scale defaults to zero for backward compat.
    if "hope_memory_w_o_init_scale" not in model_config_kwargs:
        model_config_kwargs["hope_memory_w_o_init_scale"] = 0.0
    # Hope/NL Stage 2: per-token learned-gate memory module + caps + init biases.
    # Defaults preserve Stage 1 behavior on pre-Stage-2 checkpoints.
    if "hope_memory_kind" not in model_config_kwargs:
        model_config_kwargs["hope_memory_kind"] = "linear"
    if "hope_memory_alpha_max" not in model_config_kwargs:
        model_config_kwargs["hope_memory_alpha_max"] = 0.999
    if "hope_memory_eta_max" not in model_config_kwargs:
        model_config_kwargs["hope_memory_eta_max"] = 1.0
    if "hope_memory_alpha_init_bias" not in model_config_kwargs:
        model_config_kwargs["hope_memory_alpha_init_bias"] = 4.595
    if "hope_memory_eta_init_bias" not in model_config_kwargs:
        model_config_kwargs["hope_memory_eta_init_bias"] = -2.197

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def assert_checkpoint_dir_safe(checkpoint_dir, force_overwrite=False, resume_from_step=-1):
    """Pre-flight check to prevent silently overwriting an existing trained checkpoint.

    The default `--model-tag` is `d<depth>` (e.g. `d6`), which is also where the
    canonical baseline lives. Forgetting `--model-tag` for an architectural
    variant silently overwrites the baseline after a multi-hour run. This check
    runs at training startup (before any compute) and aborts cleanly.

    - If `--resume-from-step >= 0`: assume intentional continuation, skip check.
    - If dir doesn't exist or has no `model_*.pt`: OK.
    - Otherwise: print error and `sys.exit(1)` unless `--force-overwrite`.

    Note: SFT and RL share this footgun via `chatsft_checkpoints/<model_tag>/` and
    `chatrl_checkpoints/<model_tag>/`; call this from any training entry point
    that resolves a `<model_tag>` checkpoint dir.
    """
    if resume_from_step is not None and resume_from_step >= 0:
        return
    if not os.path.isdir(checkpoint_dir):
        return
    existing = sorted(f for f in os.listdir(checkpoint_dir) if re.match(r"model_\d+\.pt$", f))
    if not existing:
        return
    if force_overwrite:
        log0(f"WARNING: --force-overwrite — proceeding despite existing checkpoint(s) in {checkpoint_dir}: {existing}")
        return
    msg = (
        f"\n\nABORT: target checkpoint directory already contains trained model(s):\n"
        f"  {checkpoint_dir}\n"
        f"  existing: {existing}\n\n"
        f"Continuing would silently overwrite. To proceed:\n"
        f"  --model-tag=<descriptive-name>  (writes to a fresh directory; recommended)\n"
        f"  --force-overwrite               (intentionally replace the existing checkpoint)\n"
        f"  --resume-from-step=<N>          (resume training from an existing step in this dir)\n\n"
    )
    print(msg, file=sys.stderr, flush=True)
    sys.exit(1)


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0, keep_last_n=None):
    """Save a (model, optimizer, meta) checkpoint triple at `step` into `checkpoint_dir`.

    Optional rolling cleanup via `keep_last_n`: after the new triple is written,
    delete older intermediate triples so only the most recent N step values remain
    on disk. Disk-constrained machines (M2, etc.) should pass keep_last_n=2 to
    cap intermediate-checkpoint footprint when --save-every is in use; without it,
    a 5000-iter pretrain with --save-every=200 leaves ~25 × 800 MB = 20 GB of
    intermediates lying around.
    """
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    # Rolling cleanup: keep only the last keep_last_n step values on disk.
    # Run on rank 0 only — all ranks share the dir, but only one needs to delete.
    if rank == 0 and keep_last_n is not None and keep_last_n >= 1:
        _trim_old_checkpoints(checkpoint_dir, keep=keep_last_n)


def _trim_old_checkpoints(checkpoint_dir, keep):
    """Keep only the `keep` most recent step values; delete older model/optim/meta files."""
    steps = set()
    for name in os.listdir(checkpoint_dir):
        m = re.match(r"(?:model|meta|optim)_(\d+)(?:_rank\d+)?\.(?:pt|json)$", name)
        if m:
            steps.add(int(m.group(1)))
    keep_steps = set(sorted(steps, reverse=True)[:keep])
    for s in steps - keep_steps:
        for name in os.listdir(checkpoint_dir):
            if re.match(rf"(?:model|meta|optim)_{s:06d}(?:_rank\d+)?\.(?:pt|json)$", name):
                os.remove(os.path.join(checkpoint_dir, name))
        log0(f"Trimmed old checkpoint at step {s} from {checkpoint_dir}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
