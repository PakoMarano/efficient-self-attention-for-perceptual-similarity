import argparse
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset import DistillImageDataset, TwoAFCDataset, load_split_paths
from model import dreamsim
from utils import log_result


def _load_teacher_embeddings(path: str) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"].float()
    paths = list(payload["paths"])
    config = payload.get("config", {})

    if embeddings.shape[0] != len(paths):
        raise ValueError(
            f"Embeddings/paths mismatch: {embeddings.shape[0]} vs {len(paths)}"
        )

    return embeddings, paths, config


def _build_teacher_map(embeddings: torch.Tensor, paths: List[str]) -> Dict[str, torch.Tensor]:
    return {path: embeddings[idx] for idx, path in enumerate(paths)}


def _prepare_for_loss(embeds: torch.Tensor) -> torch.Tensor:
    if embeds.ndim > 2:
        return embeds.reshape(embeds.shape[0], -1)
    return embeds


def _set_trainable_params(
    model: nn.Module, train_mlp: bool = True, train_norm: bool = True
) -> int:
    model.requires_grad_(False)

    vit = model.extractor.model
    trainable = 0

    for block in vit.blocks:
        if train_mlp:
            for param in block.mlp.parameters():
                param.requires_grad = True
        if train_norm:
            for param in block.norm1.parameters():
                param.requires_grad = True
            for param in block.norm2.parameters():
                param.requires_grad = True

    if train_norm:
        for param in vit.norm.parameters():
            param.requires_grad = True

    for param in model.parameters():
        if param.requires_grad:
            trainable += param.numel()

    return trainable


def _evaluate_similarity(
    dataloader: DataLoader,
    model: nn.Module,
    device: str,
    loss_type: str,
    normalize_for_loss: bool,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_device = "cuda" if use_cuda else "cpu"

    if loss_type == "cosine":
        loss_fn = nn.CosineEmbeddingLoss()
    elif loss_type == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("loss_type must be 'cosine' or 'mse'.")

    total_loss = 0.0
    total_cos = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, teacher_embed, _, _, _ in dataloader:
            if max_batches is not None and total_samples >= max_batches * dataloader.batch_size:
                break

            imgs = imgs.to(device, non_blocking=True)
            teacher_embed = teacher_embed.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device, enabled=use_cuda):
                student_embed = model.embed(imgs)
                student_embed = _prepare_for_loss(student_embed)
                teacher_embed = _prepare_for_loss(teacher_embed)

                if normalize_for_loss:
                    student_embed = F.normalize(student_embed, dim=-1)
                    teacher_embed = F.normalize(teacher_embed, dim=-1)

                if loss_type == "cosine":
                    target = torch.ones(student_embed.shape[0], device=device)
                    loss = loss_fn(student_embed, teacher_embed, target)
                else:
                    loss = loss_fn(student_embed, teacher_embed)

                cos = F.cosine_similarity(student_embed, teacher_embed, dim=-1).mean()

            batch_size_actual = imgs.shape[0]
            total_samples += batch_size_actual
            total_loss += loss.item() * batch_size_actual
            total_cos += cos.item() * batch_size_actual

    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_cosine": total_cos / max(total_samples, 1),
        "val_samples": float(total_samples),
    }


def _evaluate_2afc(
    dataloader: DataLoader,
    model: nn.Module,
    device: str,
    warmup_batches: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_device = "cuda" if use_cuda else "cpu"

    correct = 0
    total = 0
    measured_images = 0
    measured_batches = 0
    model_time = 0.0

    with torch.no_grad():
        iterator = iter(dataloader)
        for _ in range(min(warmup_batches, len(dataloader))):
            try:
                ref, left, right, _, _ = next(iterator)
            except StopIteration:
                break
            ref = ref.to(device, non_blocking=True)
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            with torch.amp.autocast(amp_device, enabled=use_cuda):
                _ = model(ref, left)
                _ = model(ref, right)

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for ref, left, right, pref_right, _ in dataloader:
            if max_batches is not None and measured_batches >= max_batches:
                break

            ref = ref.to(device, non_blocking=True)
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            target = (pref_right >= 0.5).long().to(device, non_blocking=True)

            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.amp.autocast(amp_device, enabled=use_cuda):
                dist_left = model(ref, left)
                dist_right = model(ref, right)
                pred = (dist_right < dist_left).long()
            if use_cuda:
                torch.cuda.synchronize()
            model_time += time.perf_counter() - start

            correct += (pred == target).sum().item()
            n = target.numel()
            total += n
            measured_images += n * 3
            measured_batches += 1

    accuracy = correct / max(total, 1)
    img_per_sec = measured_images / max(model_time, 1e-9)
    max_mem_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if use_cuda else 0.0

    return {
        "accuracy_2afc": accuracy,
        "img_per_sec_2afc": img_per_sec,
        "max_mem_mb_2afc": max_mem_mb,
        "samples_2afc": total,
        "batches_2afc": measured_batches,
        "timed_seconds_2afc": model_time,
    }


def run_distillation(
    teacher_embeddings: str,
    dataset_root: Optional[str] = None,
    cache_dir: str = "./models",
    output_dir: str = "./training/checkpoints",
    results_csv: str = "./reports/distill_runs.csv",
    train_split: Optional[str] = None,
    val_split: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: Optional[int] = None,
    device: str = "cuda",
    pretrained: bool = True,
    normalize_embeds: Optional[bool] = None,
    use_patch_model: Optional[bool] = None,
    unique_only: Optional[bool] = None,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    warmup_batches: int = 10,
    max_batches: Optional[int] = None,
    val_max_batches: Optional[int] = None,
    loss_type: str = "cosine",
    normalize_for_loss: bool = False,
    train_mlp: bool = True,
    train_norm: bool = True,
    save_every: int = 1,
    eval_2afc_every: int = 0,
    eval_2afc_split: str = "val",
    eval_2afc_batch_size: int = 16,
    eval_2afc_num_workers: int = 0,
    eval_2afc_max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    teacher_embeds, teacher_paths, teacher_cfg = _load_teacher_embeddings(teacher_embeddings)
    teacher_map = _build_teacher_map(teacher_embeds, teacher_paths)

    dataset_root = dataset_root or teacher_cfg.get("dataset_root", "./nights")
    img_size = img_size or teacher_cfg.get("img_size", 224)
    train_split = train_split or "train"
    val_split = val_split or "val"
    if normalize_embeds is None:
        normalize_embeds = bool(teacher_cfg.get("normalize_embeds", True))
    if use_patch_model is None:
        use_patch_model = bool(teacher_cfg.get("use_patch_model", False))
    if unique_only is None:
        unique_only = bool(teacher_cfg.get("unique_only", True))

    train_paths = load_split_paths(dataset_root, train_split)
    val_paths = load_split_paths(dataset_root, val_split)

    train_dataset = DistillImageDataset(
        dataset_root=dataset_root,
        split=train_split,
        img_size=img_size,
        teacher_map=teacher_map,
        unique_only=unique_only,
        allowed_paths=train_paths,
    )

    if len(train_dataset) == 0:
        raise ValueError("No matching samples between dataset and teacher embeddings.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataset = DistillImageDataset(
        dataset_root=dataset_root,
        split=val_split,
        img_size=img_size,
        teacher_map=teacher_map,
        unique_only=unique_only,
        allowed_paths=val_paths,
    )

    if len(val_dataset) == 0:
        raise ValueError("No validation samples found for distillation.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model, _ = dreamsim(
        pretrained=pretrained,
        device=device,
        cache_dir=cache_dir,
        normalize_embeds=normalize_embeds,
        use_patch_model=use_patch_model,
        attention_module="pool",
    )

    trainable_params = _set_trainable_params(model, train_mlp=train_mlp, train_norm=train_norm)
    if trainable_params == 0:
        raise ValueError("No trainable parameters selected. Enable MLP and/or LayerNorm.")

    model.train()

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_device = "cuda" if use_cuda else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    if loss_type == "cosine":
        loss_fn = nn.CosineEmbeddingLoss()
    elif loss_type == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("loss_type must be 'cosine' or 'mse'.")

    os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")
    total_steps = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        measured_batches = 0
        model_time = 0.0

        with torch.no_grad():
            iterator = iter(train_loader)
            for _ in range(min(warmup_batches, len(train_loader))):
                try:
                    imgs, _, _, _, _ = next(iterator)
                except StopIteration:
                    break
                imgs = imgs.to(device, non_blocking=True)
                with torch.amp.autocast(amp_device, enabled=use_cuda):
                    _ = model.embed(imgs)

        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        for imgs, teacher_embed, _, _, _ in train_loader:
            if max_batches is not None and measured_batches >= max_batches:
                break

            imgs = imgs.to(device, non_blocking=True)
            teacher_embed = teacher_embed.to(device, non_blocking=True)

            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=use_cuda):
                student_embed = model.embed(imgs)
                student_embed = _prepare_for_loss(student_embed)
                teacher_embed = _prepare_for_loss(teacher_embed)

                if normalize_for_loss:
                    student_embed = F.normalize(student_embed, dim=-1)
                    teacher_embed = F.normalize(teacher_embed, dim=-1)

                if loss_type == "cosine":
                    target = torch.ones(student_embed.shape[0], device=device)
                    loss = loss_fn(student_embed, teacher_embed, target)
                else:
                    loss = loss_fn(student_embed, teacher_embed)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if use_cuda:
                torch.cuda.synchronize()
            model_time += time.perf_counter() - start

            batch_size_actual = imgs.shape[0]
            epoch_samples += batch_size_actual
            epoch_loss += loss.item() * batch_size_actual
            measured_batches += 1
            total_steps += 1

        avg_loss = epoch_loss / max(epoch_samples, 1)
        img_per_sec = epoch_samples / max(model_time, 1e-9)
        max_mem_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if use_cuda else 0.0

        val_metrics = _evaluate_similarity(
            dataloader=val_loader,
            model=model,
            device=device,
            loss_type=loss_type,
            normalize_for_loss=normalize_for_loss,
            max_batches=val_max_batches,
        )

        record = {
            "epoch": epoch,
            "train_split": train_split,
            "val_split": val_split,
            "samples": epoch_samples,
            "batches": measured_batches,
            "loss": round(avg_loss, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_cosine": round(val_metrics["val_cosine"], 6),
            "val_samples": int(val_metrics["val_samples"]),
            "img_per_sec": round(img_per_sec, 3),
            "max_mem_mb": round(max_mem_mb, 3),
            "timed_seconds": round(model_time, 6),
            "attention_module": "pool",
            "loss_type": loss_type,
            "normalize_for_loss": normalize_for_loss,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "img_size": img_size,
            "device": device,
            "pretrained": pretrained,
            "normalize_embeds": normalize_embeds,
            "use_patch_model": use_patch_model,
            "unique_only": unique_only,
            "train_mlp": train_mlp,
            "train_norm": train_norm,
            "teacher_embeddings": teacher_embeddings,
            "lr": lr,
            "weight_decay": weight_decay,
        }

        if eval_2afc_every > 0 and (epoch % eval_2afc_every == 0):
            eval_dataset = TwoAFCDataset(
                root_dir=dataset_root,
                split=eval_2afc_split,
                load_size=img_size,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=eval_2afc_batch_size,
                shuffle=False,
                num_workers=eval_2afc_num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            eval_metrics = _evaluate_2afc(
                dataloader=eval_loader,
                model=model,
                device=device,
                warmup_batches=warmup_batches,
                max_batches=eval_2afc_max_batches,
            )
            record.update(
                {
                    "accuracy_2afc": round(eval_metrics["accuracy_2afc"], 6),
                    "img_per_sec_2afc": round(eval_metrics["img_per_sec_2afc"], 3),
                    "max_mem_mb_2afc": round(eval_metrics["max_mem_mb_2afc"], 3),
                    "samples_2afc": eval_metrics["samples_2afc"],
                    "batches_2afc": eval_metrics["batches_2afc"],
                    "timed_seconds_2afc": round(eval_metrics["timed_seconds_2afc"], 6),
                    "eval_2afc_split": eval_2afc_split,
                }
            )
        log_result(results_csv, record)

        print(
            f"epoch={epoch} loss={avg_loss:.6f} "
            f"val_loss={val_metrics['val_loss']:.6f} "
            f"val_cos={val_metrics['val_cosine']:.4f} "
            f"samples={epoch_samples} "
            f"{img_per_sec:.1f} img/s "
            f"max_mem={max_mem_mb:.1f} MB"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(output_dir, "pool_distill_best.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "config": record,
                },
                best_path,
            )

        if save_every > 0 and (epoch % save_every == 0):
            ckpt_path = os.path.join(output_dir, f"pool_distill_epoch_{epoch}.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "config": record,
                },
                ckpt_path,
            )

    return {
        "best_loss": best_loss,
        "epochs": epochs,
        "steps": total_steps,
        "output_dir": output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill pool attention model using saved teacher embeddings."
    )
    parser.add_argument("--teacher_embeddings", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--output_dir", type=str, default="./training/checkpoints")
    parser.add_argument("--results_csv", type=str, default="./reports/distill_runs.csv")
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--val_max_batches", type=int, default=None)
    parser.add_argument("--loss_type", type=str, default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--normalize_for_loss", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_normalize_embeds", action="store_true")
    parser.add_argument("--use_patch_model", action="store_true")
    parser.add_argument("--keep_duplicates", action="store_true")
    parser.add_argument("--no_train_mlp", action="store_true")
    parser.add_argument("--no_train_norm", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--eval_2afc_every", type=int, default=0)
    parser.add_argument("--eval_2afc_split", type=str, default="val")
    parser.add_argument("--eval_2afc_batch_size", type=int, default=16)
    parser.add_argument("--eval_2afc_num_workers", type=int, default=0)
    parser.add_argument("--eval_2afc_max_batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    run_distillation(
        teacher_embeddings=args.teacher_embeddings,
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        device=args.device,
        pretrained=not args.no_pretrained,
        normalize_embeds=False if args.no_normalize_embeds else None,
        use_patch_model=True if args.use_patch_model else None,
        unique_only=not args.keep_duplicates,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_batches=args.warmup_batches,
        max_batches=args.max_batches,
        val_max_batches=args.val_max_batches,
        loss_type=args.loss_type,
        normalize_for_loss=args.normalize_for_loss,
        train_mlp=not args.no_train_mlp,
        train_norm=not args.no_train_norm,
        save_every=args.save_every,
        eval_2afc_every=args.eval_2afc_every,
        eval_2afc_split=args.eval_2afc_split,
        eval_2afc_batch_size=args.eval_2afc_batch_size,
        eval_2afc_num_workers=args.eval_2afc_num_workers,
        eval_2afc_max_batches=args.eval_2afc_max_batches,
    )


if __name__ == "__main__":
    main()
