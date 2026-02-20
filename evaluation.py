import argparse
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from dataset import TwoAFCDataset
from model import dreamsim
from utils import log_result


def create_dataloader(
    dataset_root: str,
    split: str = "test",
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
) -> DataLoader:
    dataset = TwoAFCDataset(root_dir=dataset_root, split=split, load_size=img_size)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0 when provided.")
        max_samples = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def eval_2afc(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: str = "cuda",
    warmup_batches: int = 10,
    max_batches: Optional[int] = None,
):
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
                ref, left, right, pref_right, _ = next(iterator)
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
        "img_per_sec": img_per_sec,
        "max_mem_mb": max_mem_mb,
        "samples": total,
        "batches": measured_batches,
        "warmup_batches": min(warmup_batches, len(dataloader)),
        "timed_seconds": model_time,
    }


def run_evaluation(
    dataset_root: str,
    cache_dir: str,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = 224,
    device: str = "cuda",
    pretrained: bool = True,
    normalize_embeds: bool = True,
    use_patch_model: bool = False,
    warmup_batches: int = 10,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    impl: str = "benchmark",
    results_csv: str = "./reports/results.csv",
):
    dataloader = create_dataloader(
        dataset_root=dataset_root,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )

    model, _ = dreamsim(
        pretrained=pretrained,
        device=device,
        cache_dir=cache_dir,
        normalize_embeds=normalize_embeds,
        use_patch_model=use_patch_model,
    )

    metrics = eval_2afc(
        dataloader=dataloader,
        model=model,
        device=device,
        warmup_batches=warmup_batches,
        max_batches=max_batches,
    )

    record = {
        "split": split,
        "impl": impl,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "img_size": img_size,
        "use_patch_model": use_patch_model,
        "normalize_embeds": normalize_embeds,
        "pretrained": pretrained,
        "max_batches": max_batches,
        "max_samples": max_samples,
        "accuracy_2afc": round(metrics["accuracy_2afc"], 6),
        "img_per_sec": round(metrics["img_per_sec"], 3),
        "max_mem_mb": round(metrics["max_mem_mb"], 3),
        "samples": metrics["samples"],
        "batches": metrics["batches"],
        "warmup_batches": metrics["warmup_batches"],
        "timed_seconds": round(metrics["timed_seconds"], 6),
    }
    log_result(results_csv, record)

    print(
        f"split={split}, impl={impl}: "
        f"acc={metrics['accuracy_2afc']:.4f}, "
        f"{metrics['img_per_sec']:.1f} img/s, "
        f"max_mem={metrics['max_mem_mb']:.1f} MB"
    )

    return record


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DreamSim model on 2AFC dataset.")
    parser.add_argument("--dataset_root", type=str, default="./nights")
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--results_csv", type=str, default="./reports/results.csv")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--impl", type=str, default="benchmark")
    parser.add_argument("--use_patch_model", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_normalize_embeds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    run_evaluation(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        device=args.device,
        pretrained=not args.no_pretrained,
        normalize_embeds=not args.no_normalize_embeds,
        use_patch_model=args.use_patch_model,
        warmup_batches=args.warmup_batches,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        impl=args.impl,
        results_csv=args.results_csv,
    )


if __name__ == "__main__":
    main()
