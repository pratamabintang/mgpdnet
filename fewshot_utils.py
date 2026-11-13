"""
Shared helpers for the few-shot landslide segmentation train/test scripts.

Kept dependency-light (no sacred / no config.py) so the scripts run standalone.
"""

import logging
import os
import platform
import random
import shutil
import sys
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

try:
    import torchvision
except Exception:  # pragma: no cover
    torchvision = None

from models.fewshot import FewShotSeg


# --------------------------------------------------------------------------- #
# Logging (lightweight; only at stage boundaries)
# --------------------------------------------------------------------------- #

_LOGGER_NAME = "mgpdnet"


def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a stdout logger (optionally also file), configured once."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        parent = os.path.dirname(os.path.abspath(log_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _ram_gb() -> Optional[Tuple[float, float]]:
    """(free, total) RAM in GB, or None if unavailable."""
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        return vm.available / 2**30, vm.total / 2**30
    except Exception:
        pass
    try:
        info: Dict[str, float] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if rest:
                    info[key.strip()] = float(rest.split()[0])
        free = info.get("MemAvailable", info.get("MemFree", 0.0))
        total = info.get("MemTotal", 0.0)
        return free / 2**20, total / 2**20
    except Exception:
        return None


def log_system(logger: logging.Logger) -> None:
    """One-shot snapshot of the machine so we can tell whether it is alive."""
    logger.info(
        "system | python=%s | %s %s | pid=%d",
        sys.version.split()[0], platform.system(), platform.release(), os.getpid(),
    )
    logger.info(
        "torch  | torch=%s | cuda_available=%s | cuda=%s | cudnn=%s",
        torch.__version__, torch.cuda.is_available(), torch.version.cuda,
        torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    )
    logger.info("cpu    | count=%s", os.cpu_count())
    ram = _ram_gb()
    if ram is not None:
        logger.info("ram    | free=%.1fGB total=%.1fGB", *ram)


def log_gpu(logger: logging.Logger) -> None:
    if not torch.cuda.is_available():
        logger.info("gpu    | none (CPU run)")
        return
    logger.info("gpu    | count=%d", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
            mem = f"mem={free / 2**30:.1f}/{total / 2**30:.1f}GB"
        except Exception:
            mem = ""
        logger.info(
            "gpu[%d] | %s sm_%d%d %s bf16=%s",
            i, props.name, cap[0], cap[1], mem, bf16_supported(),
        )


def log_gpu_mem(logger: logging.Logger, tag: str = "") -> None:
    """Cheap CUDA memory report (no sync); used at stage boundaries only."""
    if not torch.cuda.is_available():
        return
    try:
        alloc = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "gpu_mem%s | allocated=%.2fGB reserved=%.2fGB free=%.1f/%.1fGB",
            f"[{tag}]" if tag else "", alloc, reserved, free / 2**30, total / 2**30,
        )
    except Exception:
        pass


def _existing_dir(path: str) -> str:
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def log_disk(logger: logging.Logger, *paths: str) -> None:
    seen = set()
    for path in paths:
        if not path:
            continue
        base = _existing_dir(path)
        if base in seen:
            continue
        seen.add(base)
        try:
            u = shutil.disk_usage(base)
            logger.info("disk   | %s free=%.1fGB total=%.1fGB", base,
                        u.free / 2**30, u.total / 2**30)
        except Exception:
            pass


def log_config(logger: logging.Logger, args) -> None:
    logger.info("--- parameters ---")
    for k in sorted(vars(args)):
        logger.info("  %-22s = %r", k, getattr(args, k))


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #


def seed_everything(seed: int = 2022) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def bf16_supported() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:  # pragma: no cover
        return False


def resolve_amp(mode: str, device: torch.device):
    """Return (use_amp, dtype).

    ``auto`` picks bf16 when the GPU supports it (Ampere or newer), otherwise
    fp16. No-op on CPU or mode='none'.
    """
    if mode in ("none", "False", "0", "") or device.type != "cuda":
        return False, torch.float32
    if mode == "fp16":
        dtype = torch.float16
    elif mode == "bf16":
        dtype = torch.bfloat16
        if not bf16_supported():
            print("[warn] AMP=bf16 requested but this GPU has no bf16 support; "
                  "results may be slow/imprecise. Consider AMP=auto or fp16.")
    else:  # auto
        dtype = torch.bfloat16 if bf16_supported() else torch.float16
    return True, dtype


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    try:
        idx = device.index if device.index is not None else 0
        cap = torch.cuda.get_device_capability(idx)
        return (f"{torch.cuda.get_device_name(idx)} (sm_{cap[0]}{cap[1]}, "
                f"bf16={bf16_supported()})")
    except Exception:  # pragma: no cover
        return str(device)


# --------------------------------------------------------------------------- #
# Pretrained backbone weights
# --------------------------------------------------------------------------- #


def _download_deeplabv3_resnet101() -> Dict[str, torch.Tensor]:
    """torchvision DeepLabV3-ResNet101 COCO weights (keys already 'backbone.*')."""
    weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
    return model.state_dict()


def _download_resnet101() -> Dict[str, torch.Tensor]:
    """torchvision ResNet101 ImageNet weights, remapped to the 'backbone.*' keys
    that ``Res101Encoder`` uses."""
    weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet101(weights=weights)
    sd = model.state_dict()
    return {f"backbone.{k}": v for k, v in sd.items()}


def load_pretrained(
    pretrained: str = "deeplabv3", ckpt_dir: str = "checkpoints"
) -> Optional[Dict[str, torch.Tensor]]:
    """Return a state_dict dict for the backbone, or None.

    ``pretrained``: 'deeplabv3' | 'resnet101' | 'none' | path to a .pth file.
    """
    if pretrained in (None, "", "none"):
        return None

    if os.path.isfile(pretrained):
        return torch.load(pretrained, map_location="cpu")

    os.makedirs(ckpt_dir, exist_ok=True)
    local = os.path.join(ckpt_dir, os.path.basename(pretrained) + ".pth")
    if os.path.isfile(local):
        return torch.load(local, map_location="cpu")

    if torchvision is None:
        print("[warn] torchvision unavailable; training from random init.")
        return None

    try:
        if pretrained == "deeplabv3":
            sd = _download_deeplabv3_resnet101()
        elif pretrained == "resnet101":
            sd = _download_resnet101()
        else:
            print(f"[warn] unknown pretrained='{pretrained}'; random init.")
            return None
        torch.save(sd, local)
        print(f"[info] cached pretrained weights -> {local}")
        return sd
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[warn] could not download pretrained weights ({exc}); random init.")
        return None


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_model(
    pretrained="deeplabv3",
    device: Optional[torch.device] = None,
    ckpt_dir: str = "checkpoints",
) -> FewShotSeg:
    """Build the model.

    ``pretrained`` may be a scheme string handled by :func:`load_pretrained`
    ('deeplabv3' | 'resnet101' | 'none' | path) or an already-loaded state_dict
    dict (e.g. obtained by calling :func:`load_pretrained` explicitly).
    """
    device = device or resolve_device()
    if isinstance(pretrained, dict) or pretrained is None:
        weights = pretrained
    else:
        weights = load_pretrained(pretrained, ckpt_dir)
    model = FewShotSeg(pretrained_weights=weights if weights is not None else None)
    # ``FewShotSeg`` hard-codes cuda; make it follow the actual device.
    model.device = device
    model = model.to(device)
    # Non-registered constant used in loss math; keep it on the right device.
    model.alpha = model.alpha.to(device)
    return model


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _set_stats(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def task_loss(output: torch.Tensor, gt: torch.Tensor) -> float:
    """Query segmentation loss (NLL on the 2-channel probabilities).

    Mirrors the ``loss_qry`` term used in training so the value is comparable.
    ``output``: float probs [B, 2, H, W]; ``gt``: {0,1} [B, H, W].
    """
    gt = gt.long()
    label = torch.full_like(gt, 255)
    label[gt == 1] = 1
    label[gt == 0] = 0
    eps = torch.finfo(torch.float32).eps
    log_prob = torch.log(torch.clamp(output.float(), eps, 1 - eps))
    return float(torch.nn.functional.nll_loss(log_prob, label).item())


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Binary landslide metrics for one volume.

    ``pred``/``gt``: 2D arrays of {0,1}. DSC/IoU use the foreground class;
    episodes with empty gt *and* empty pred are counted as perfect (1.0).
    """
    tp, fp, fn, tn = _set_stats(pred, gt)
    eps = 1e-7

    if (tp + fp + fn) == 0:
        dice = iou = prec = rec = acc = 1.0
    else:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn)
        iou = tp / (tp + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(acc),
    }


def aggregate_metrics(per_episode: Iterable[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Return {metric: (mean, std)} across episodes."""
    rows = list(per_episode)
    if not rows:
        return {}
    keys = rows[0].keys()
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


def format_metrics(agg: Dict[str, Tuple[float, float]]) -> str:
    return " | ".join(f"{k}: {m * 100:.2f}+-{s * 100:.2f}" for k, (m, s) in agg.items())


def batch_dice(output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean foreground Dice over the batch. output logits [B,2,H,W], gt [B,H,W]."""
    pred = output.argmax(dim=1)
    gt = gt.long()
    dice_list = []
    for b in range(pred.shape[0]):
        tp, fp, fn, _ = _set_stats(pred[b].cpu().numpy(), gt[b].cpu().numpy())
        if (tp + fp + fn) == 0:
            dice_list.append(1.0)
        else:
            dice_list.append(2.0 * tp / (2.0 * tp + fp + fn))
    return torch.tensor(float(np.mean(dice_list)), device=output.device)
