import argparse
import logging
import os
import time

import numpy as np
import torch

from dataloaders.landslide import DEFAULT_DATA_DIR, get_dataloader
from fewshot_utils import (
    aggregate_metrics,
    batch_dice,
    build_model,
    compute_metrics,
    describe_device,
    format_metrics,
    load_pretrained,
    log_config,
    log_disk,
    log_gpu,
    log_gpu_mem,
    log_system,
    resolve_amp,
    resolve_device,
    seed_everything,
    setup_logger,
    task_loss,
    to_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MGPD-Net landslide few-shot training")
    # data
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="validation")
    p.add_argument("--normalize", default="imagenet", choices=["imagenet", "landslide", "none"])
    p.add_argument("--input-scale", type=float, default=1.0,
                   help="Multiply raw bands by this before normalization (see docs).")
    p.add_argument("--img-size", type=int, default=None)
    # augmentation (train only)
    p.add_argument("--aug-flip", action="store_true", help="Random H/V flip.")
    p.add_argument("--aug-rot90", action="store_true", help="Random k*90-degree rotation.")
    p.add_argument("--aug-rotation", type=float, default=0.0,
                   help="Max abs rotation angle in degrees (0 disables; needs cv2).")
    p.add_argument("--aug-color", action="store_true",
                   help="Photometric jitter on spectral bands only (not topography).")
    p.add_argument("--color-brightness", type=float, default=0.2)
    p.add_argument("--color-contrast", type=float, default=0.2)
    p.add_argument("--color-gamma", type=float, default=0.2)
    p.add_argument("--color-gain", type=float, default=0.1)
    # episode
    p.add_argument("--n-shot", type=int, default=1)
    p.add_argument("--n-query", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    # training
    p.add_argument("--n-steps", type=int, default=30000)
    p.add_argument("--max-iters-per-load", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1, help="Must stay 1 for MGPD-Net.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--lr-step-gamma", type=float, default=0.80)
    p.add_argument("--align-weight", type=float, default=1.0)
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--contrastive-weight", type=float, default=1.0)
    p.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "none"],
                   help="Mixed precision on CUDA (no-op on CPU).")
    # bookkeeping
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--pretrained", default="deeplabv3",
                   help="'deeplabv3' | 'resnet101' | 'none' | path to .pth")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--download-only", action="store_true",
                   help="Download/cache the pretrained backbone weights and exit.")
    p.add_argument("--output-dir", default=os.path.join("runs", "landslide_fss"))
    p.add_argument("--print-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=1000,
                   help="Quick val DSC every N steps (0 disables).")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--log-file", default=None, help="Optional log file path.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


@torch.no_grad()
def quick_eval(model, args, device, logger) -> str:
    loader = get_dataloader(
        data_dir=args.data_dir,
        split=args.val_split,
        n_shot=args.n_shot,
        n_query=args.n_query,
        batch_size=1,
        num_workers=0,
        max_iters=args.eval_episodes,
        normalize=args.normalize,
        input_scale=args.input_scale,
        img_size=args.img_size,
        seed=args.seed,
        pin_memory=False,
    )
    model.eval()
    rows = []
    losses = []
    for i, batch in enumerate(loader):
        if i >= args.eval_episodes:
            break
        batch = to_device(batch, device)
        output, *_ = model(
            batch["supp_imgs"], batch["supp_mask"], batch["qry_imgs"], batch["qry_mask"],
            train=False,
        )
        losses.append(task_loss(output, batch["qry_mask"]))
        pred = output.argmax(dim=1)[0].cpu().numpy()
        gt = batch["qry_mask"][0].cpu().numpy()
        rows.append(compute_metrics(pred, gt))
    model.train()
    val_loss = float(np.mean(losses)) if losses else float("nan")
    return f"val_loss={val_loss:.4f} | {format_metrics(aggregate_metrics(rows))}"


def main() -> None:
    args = parse_args()
    assert args.batch_size == 1, "MGPD-Net FewShotSeg only supports batch_size=1."
    logger = setup_logger(args.log_file, getattr(logging, args.log_level))

    # --- Stage 0: environment + parameters -----------------------------------
    logger.info("=== startup ===")
    log_system(logger)
    log_gpu(logger)
    log_disk(logger, args.data_dir, args.output_dir, args.ckpt_dir,
             os.environ.get("LOCAL_DATA") or "", "/content")
    log_config(logger, args)
    seed_everything(args.seed)

    # --- Stage 1: pretrained backbone weights (download + cache once) ---------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logger.info("=== stage: pretrained (fetch/cache) ===")
    t = time.time()
    weights = load_pretrained(args.pretrained, args.ckpt_dir)
    logger.info("pretrained='%s' cached=%s -> %s (%.1fs)",
                args.pretrained, weights is not None,
                os.path.abspath(args.ckpt_dir), time.time() - t)
    if args.download_only:
        logger.info("--download-only set; exiting after checkpoint fetch.")
        return

    # --- Stage 2: device / model ---------------------------------------------
    device = resolve_device(args.gpu)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("=== stage: model build ===")
    logger.info("device=%s [%s] out=%s", device, describe_device(device), args.output_dir)
    model = build_model(weights, device=device, ckpt_dir=args.ckpt_dir)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model ready | params=%.2fM", n_params / 1e6)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_step_gamma)

    use_amp, amp_dtype = resolve_amp(args.amp, device)
    scaler_enabled = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    logger.info("amp=%s -> enabled=%s dtype=%s scaler=%s",
                args.amp, use_amp, amp_dtype, scaler_enabled)
    log_gpu_mem(logger, "after-model")

    # --- Stage 3: training loop ---------------------------------------------
    step = 0
    epoch = 0
    t0 = time.time()
    running = np.zeros(5, dtype=np.float64)  # total, align, mse, qry, contrastive
    try:
        while step < args.n_steps:
            logger.info("=== stage: epoch %d (step %d/%d) ===", epoch, step, args.n_steps)
            ep_t = time.time()
            loader = get_dataloader(
                data_dir=args.data_dir,
                split=args.split,
                n_shot=args.n_shot,
                n_query=args.n_query,
                batch_size=1,
                num_workers=args.num_workers,
                max_iters=min(args.max_iters_per_load, args.n_steps - step),
                normalize=args.normalize,
                input_scale=args.input_scale,
                img_size=args.img_size,
                seed=args.seed + step,
                pin_memory=(device.type == "cuda"),
                aug_flip=args.aug_flip,
                aug_rot90=args.aug_rot90,
                aug_color=args.aug_color,
                aug_rotation=args.aug_rotation,
                color_brightness=args.color_brightness,
                color_contrast=args.color_contrast,
                color_gamma=args.color_gamma,
                color_gain=args.color_gain,
            )
            logger.info("dataset ready | split=%s pairs=%d episodes=%d workers=%d",
                        args.split, len(loader.dataset.files),
                        len(loader.dataset), args.num_workers)
            for batch in loader:
                batch = to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    output, align_loss, mse_loss, loss_qry, contrastive_loss = model(
                        batch["supp_imgs"], batch["supp_mask"], batch["qry_imgs"],
                        batch["qry_mask"], train=True,
                    )
                    total = (
                        loss_qry
                        + args.align_weight * align_loss
                        + args.mse_weight * mse_loss
                        + args.contrastive_weight * contrastive_loss
                    )

                optimizer.zero_grad(set_to_none=True)
                if scaler_enabled:
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                step += 1
                vals = np.array([
                    float(total.item()), float(align_loss.item()), float(mse_loss.item()),
                    float(loss_qry.item()), float(contrastive_loss.item()),
                ])
                running += vals

                if step % args.print_interval == 0:
                    avg = running / args.print_interval
                    with torch.no_grad():
                        dsc = batch_dice(output.detach(), batch["qry_mask"]).item()
                    logger.info(
                        "step %6d | total=%.4f align=%.4f mse=%.4f qry=%.4f "
                        "contra=%.4f train_dsc=%.2f lr=%.2e | %.1fs",
                        step, avg[0], avg[1], avg[2], avg[3], avg[4], dsc * 100,
                        optimizer.param_groups[0]["lr"], time.time() - t0,
                    )
                    running[:] = 0.0

                if args.save_interval and step % args.save_interval == 0:
                    ckpt = os.path.join(args.output_dir, f"{step}.pth")
                    torch.save({"step": step, "model": model.state_dict(),
                                "optimizer": optimizer.state_dict()}, ckpt)
                    logger.info("saved checkpoint -> %s", ckpt)
                    log_gpu_mem(logger, "save")

                if args.eval_interval and step % args.eval_interval == 0:
                    logger.info("=== stage: val @ %d ===", step)
                    logger.info("val result | %s", quick_eval(model, args, device, logger))

                if step >= args.n_steps:
                    break
            scheduler.step()
            epoch += 1
            logger.info("epoch %d done | %.1fs | lr=%.2e",
                        epoch - 1, time.time() - ep_t, optimizer.param_groups[0]["lr"])

        final = os.path.join(args.output_dir, "final.pth")
        torch.save({"step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, final)
        logger.info("=== done ===")
        logger.info("trained %d steps; saved %s", step, final)
        log_gpu_mem(logger, "final")
    except KeyboardInterrupt:
        ckpt = os.path.join(args.output_dir, f"interrupt_{step}.pth")
        torch.save({"step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt)
        logger.warning("interrupted at step %d; saved %s", step, ckpt)


if __name__ == "__main__":
    main()
