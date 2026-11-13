"""
Evaluate a trained MGPD-Net on few-shot landslide segmentation episodes.

Example:
    python test.py --checkpoint runs/landslide_fss/final.pth \
        --split validation --n-shot 1 --eval-episodes 300
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

from dataloaders.landslide import DEFAULT_DATA_DIR, get_dataloader
from fewshot_utils import (
    aggregate_metrics,
    build_model,
    compute_metrics,
    describe_device,
    format_metrics,
    log_config,
    log_disk,
    log_gpu,
    log_system,
    resolve_device,
    seed_everything,
    setup_logger,
    task_loss,
    to_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MGPD-Net landslide few-shot evaluation")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--split", default="validation",
                   choices=["validation", "val", "test", "train"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--normalize", default="imagenet", choices=["imagenet", "landslide", "none"])
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--n-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=300)
    p.add_argument("--repeat", type=int, default=1,
                   help="Resample the evaluation set this many times.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--pretrained", default="none",
                   help="Only used if the checkpoint holds no model weights.")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--output-json", default=None)
    p.add_argument("--save-vis", default=None,
                   help="Directory to dump a few prediction PNGs (needs matplotlib).")
    p.add_argument("--vis-count", type=int, default=8)
    p.add_argument("--log-file", default=None, help="Optional log file path.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def load_checkpoint(model, path: str, device) -> int:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        return int(state.get("step", -1))
    model.load_state_dict(state)
    return -1


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file, getattr(logging, args.log_level))

    logger.info("=== startup ===")
    log_system(logger)
    log_gpu(logger)
    log_disk(logger, args.data_dir, args.checkpoint,
             os.environ.get("LOCAL_DATA") or "", "/content")
    log_config(logger, args)
    seed_everything(args.seed)

    device = resolve_device(args.gpu)
    logger.info("=== stage: model build ===")
    logger.info("device=%s [%s]", device, describe_device(device))
    model = build_model(args.pretrained, device=device, ckpt_dir=args.ckpt_dir)

    logger.info("=== stage: load checkpoint ===")
    t = time.time()
    step = load_checkpoint(model, args.checkpoint, device)
    logger.info("loaded %s | step=%s (%.1fs)", args.checkpoint, step, time.time() - t)
    model.eval()

    rows = []
    saved = 0
    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)

    with torch.no_grad():
        for rep in range(args.repeat):
            logger.info("=== stage: eval rep %d/%d (split=%s, episodes=%d) ===",
                        rep + 1, args.repeat, args.split, args.eval_episodes)
            loader = get_dataloader(
                data_dir=args.data_dir,
                split=args.split,
                n_shot=args.n_shot,
                n_query=args.n_query,
                batch_size=1,
                num_workers=args.num_workers,
                max_iters=args.eval_episodes,
                normalize=args.normalize,
                input_scale=args.input_scale,
                img_size=args.img_size,
                seed=args.seed + rep * 100003,
                pin_memory=(device.type == "cuda"),
            )
            logger.info("dataset ready | pairs=%d episodes=%d workers=%d",
                        len(loader.dataset.files), len(loader.dataset), args.num_workers)
            rep_t = time.time()
            for i, batch in enumerate(loader):
                if i >= args.eval_episodes:
                    break
                batch = to_device(batch, device)
                output, *_ = model(
                    batch["supp_imgs"], batch["supp_mask"],
                    batch["qry_imgs"], batch["qry_mask"], train=False,
                )
                pred = output.argmax(dim=1)[0].cpu().numpy()
                gt = batch["qry_mask"][0].cpu().numpy()
                m = compute_metrics(pred, gt)
                m["loss"] = task_loss(output, batch["qry_mask"])
                m["query"] = batch["query_names"][0][0]
                rows.append(m)

                if args.save_vis and saved < args.vis_count:
                    _dump_vis(args, batch, pred, gt, saved, device, logger)
                    saved += 1
            logger.info("rep %d done | %d episodes | %.1fs", rep + 1,
                        args.eval_episodes, time.time() - rep_t)

    metrics_only = [{k: v for k, v in r.items() if k != "query"} for r in rows]
    agg = aggregate_metrics(metrics_only)
    logger.info("=== stage: report ===")
    logger.info("ckpt=%s step=%s split=%s n_shot=%d episodes=%d",
                args.checkpoint, step, args.split, args.n_shot, len(rows))
    logger.info("result | %s", format_metrics(agg))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"aggregate": {k: list(v) for k, v in agg.items()},
                       "per_episode": rows,
                       "args": vars(args)}, f, indent=2)
        logger.info("wrote %s", args.output_json)


def _dump_vis(args, batch, pred, gt, idx, device, logger=None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if logger:
            logger.warning("matplotlib unavailable, skip vis (%s)", exc)
        return
    img = batch["qry_imgs"][0][0].cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-7)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(img); ax[0].set_title("query RGB")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("gt")
    ax[2].imshow(pred, cmap="gray"); ax[2].set_title("pred")
    for a in ax:
        a.axis("off")
    out = os.path.join(args.save_vis, f"ep_{idx:03d}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
