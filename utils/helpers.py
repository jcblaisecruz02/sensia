import os, re, glob, random, datetime
import numpy as np
import torch
import torch.distributed as dist

from dataclasses import asdict
from importlib import import_module

def get_num_workers():
    try:
        n_cores = os.cpu_count() or 4
        return max(1, min(8, n_cores - 2))
    except:
        return 4

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["HF_DETERMINISTIC"] = "1"
    os.environ["TRANSFORMERS_SEED"] = str(seed)

    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return _worker_init_fn

def log_trainable_params(model, logger=None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = 100.0 * trainable_params / max(1, total_params)
    if logger:
        logger.info(f"[Model] Trainable parameters: {trainable_params:,} / {total_params:,} ({percent:.2f}%)")
    else:
        print(f"[Model] Trainable parameters: {trainable_params:,} / {total_params:,} ({percent:.2f}%)")


def log_param_breakdown(model, logger=None, top_k=10):
    prefer_first = ["word_embeddings", "position_embeddings", "gpt2_model"]
    children = dict(model.named_children())
    order = [n for n in prefer_first if n in children] + [n for n in children if n not in prefer_first]

    seen = set(); counts = {}
    for name in order:
        m = children[name]
        cnt = 0
        for p in m.parameters(recurse=True):
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); cnt += p.numel()
        counts[name] = cnt
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    lines = [f"    {n}: {c:,}" for n, c in top]
    (logger.info if logger else print)("[Model] Top trainable modules (unique):\n" + "\n".join(lines))

def _checkpoint_dir(save_dir: str) -> str:
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir

def _checkpoint_name(step: int) -> str:
    return f"step-{int(step):012d}.pt"

def _list_checkpoints(save_dir: str):
    """Return list of (path, step) sorted by step ascending."""
    ckpt_dir = _checkpoint_dir(save_dir)
    paths = glob.glob(os.path.join(ckpt_dir, "step-*.pt"))
    items = []
    for p in paths:
        m = re.search(r"step-(\d+)\.pt$", p)
        if m:
            items.append((p, int(m.group(1))))
    items.sort(key=lambda x: x[1])
    return items

def save_checkpoint(
    save_dir: str,
    step: int,
    *,
    model,
    optimizer,
    lr_scheduler=None,
    scaler=None,
    context_loss_fn=None,
    sense_loss_fn=None,
    loss_scheduler=None,
    args=None,
    extra_state: dict | None = None,
    max_checkpoints: int = 0,
    logger=None,
):
    """Save a full-fidelity checkpoint capturing the entire training run state."""
    ckpt_dir = _checkpoint_dir(save_dir)
    tmp_path = os.path.join(ckpt_dir, _checkpoint_name(step) + ".tmp")
    final_path = os.path.join(ckpt_dir, _checkpoint_name(step))

    state = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "context_loss_fn": {
            "temperature": getattr(context_loss_fn, "temperature", None)
        } if context_loss_fn is not None else None,
        "sense_loss_fn": {
            "temperature": getattr(sense_loss_fn, "temperature", None)
        } if sense_loss_fn is not None else None,
        "loss_scheduler": {
            "cfg": asdict(loss_scheduler.cfg) if hasattr(loss_scheduler, "cfg") else None,
        } if loss_scheduler is not None else None,
        "args": vars(args) if args is not None else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra_state": extra_state or {},
    }

    # Atomic-ish save
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)

    # Retention policy
    if (max_checkpoints or 0) > 0:
        items = _list_checkpoints(save_dir)
        while len(items) > max_checkpoints:
            old_path, old_step = items.pop(0)
            try:
                os.remove(old_path)
                if logger: logger.info(f"[ckpt] Pruned {os.path.basename(old_path)}")
            except Exception as e:
                if logger: logger.warning(f"[ckpt] Failed to prune {old_path}: {e}")

    if logger: logger.info(f"[ckpt] Saved {os.path.basename(final_path)}")

def load_latest_checkpoint(
    save_dir: str,
    *,
    model,
    optimizer=None,
    lr_scheduler=None,
    scaler=None,
    context_loss_fn=None,
    sense_loss_fn=None,
    loss_scheduler=None,
    logger=None,
):
    """Load the most recent checkpoint. Returns (loaded: bool, step: int, state: dict)."""
    items = _list_checkpoints(save_dir)
    if not items:
        if logger: logger.info("[ckpt] No checkpoints found.")
        return False, 0, {}

    path, step = items[-1]

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            local_device = torch.device(f"cuda:{local_rank}")
        else:
            local_device = torch.device("cpu")
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(local_device, non_blocking=True)
    
    # --- LR scheduler / scaler ---
    if lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    # --- Restore InfoNCE temps if present ---
    if context_loss_fn is not None and ckpt.get("context_loss_fn"):
        t = ckpt["context_loss_fn"].get("temperature")
        if t is not None:
            context_loss_fn.temperature = float(t)
    if sense_loss_fn is not None and ckpt.get("sense_loss_fn"):
        t = ckpt["sense_loss_fn"].get("temperature")
        if t is not None:
            sense_loss_fn.temperature = float(t)

    # Restore RNG
    rng = ckpt.get("rng_state", {})
    try:
        if "python" in rng and rng["python"] is not None: random.setstate(rng["python"])
        if "numpy" in rng and rng["numpy"] is not None: np.random.set_state(rng["numpy"])
        if "torch" in rng and rng["torch"] is not None: torch.set_rng_state(rng["torch"])
        if "cuda_all" in rng and rng["cuda_all"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda_all"])
    except Exception as e:
        if logger: logger.warning(f"[ckpt] Could not fully restore RNG state: {e}")

    if logger: logger.info(f"[ckpt] Loaded {os.path.basename(path)} (step={step})")
    return True, int(step), ckpt

def maybe_resume(
    args,
    model,
    optimizer,
    lr_scheduler,
    scaler,
    context_loss_fn,
    sense_loss_fn,
    loss_scheduler,
    logger=None,
):
    """
    If args.resume_from_checkpoint is set, load the latest checkpoint and return starting step.
    Returns (start_step, extra_state_dict).
    """
    if not getattr(args, "resume_from_checkpoint", False):
        return 0, {}

    ok, step, ckpt = load_latest_checkpoint(
        args.save_dir,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        context_loss_fn=context_loss_fn,
        sense_loss_fn=sense_loss_fn,
        loss_scheduler=loss_scheduler,
        logger=logger,
    )
    if not ok:
        return 0, {}

    # let the caller fast-forward progress bars and logs
    return int(step), ckpt.get("extra_state", {})

def maybe_save(
    args,
    step: int,
    *,
    model,
    optimizer,
    lr_scheduler,
    scaler,
    context_loss_fn,
    sense_loss_fn,
    loss_scheduler,
    extra_state: dict | None = None,
    logger=None,
):
    if not is_main_process():
        return
    
    """Save a checkpoint every args.save_every steps with retention."""
    save_every = int(getattr(args, "save_every", 0) or 0)
    if save_every <= 0:
        return
    if step % save_every != 0:
        return
    max_ckpts = int(getattr(args, "max_checkpoints", 0) or 0)

    to_save = model.module if hasattr(model, "module") else model

    save_checkpoint(
        args.save_dir,
        step,
        model=to_save,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        context_loss_fn=context_loss_fn,
        sense_loss_fn=sense_loss_fn,
        loss_scheduler=loss_scheduler,
        args=args,
        extra_state=extra_state,
        max_checkpoints=max_ckpts,
        logger=logger,
    )

def dist_is_available_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if dist_is_available_and_initialized() else 1

def get_rank():
    return dist.get_rank() if dist_is_available_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def barrier():
    if dist_is_available_and_initialized():
        dist.barrier()

def init_distributed(logger=None, backend="nccl", timeout_seconds=1800):
    if dist.is_available() and dist.is_initialized():
        return

    if "RANK" not in os.environ:
        if logger: logger.info("[DDP] No torch.distributed env found; running single-process.")
        return

    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)

    # Pass device_id if supported (PyTorch >= 2.7)
    kwargs = dict(backend=backend, init_method="env://",
                  timeout=datetime.timedelta(seconds=timeout_seconds))
    try:
        # Some versions accept `device_id` (int) or list[int]
        kwargs["device_id"] = local_rank
        dist.init_process_group(**kwargs)
    except TypeError:
        # Fallback for older versions
        kwargs.pop("device_id", None)
        dist.init_process_group(**kwargs)

    if logger:
        logger.info(f"[DDP] Initialized rank {rank}/{world_size} on device cuda:{local_rank}")

def cleanup_distributed():
    if dist_is_available_and_initialized():
        dist.destroy_process_group()
