import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader, Subset

from dataset import SingleImageDataset
from model import dreamsim
from utils import log_result


def create_single_image_dataloader(
    dataset_root: str,
    split: str = "all",
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    unique_only: bool = True,
) -> DataLoader:
    dataset = SingleImageDataset(
        root_dir=dataset_root,
        split=split,
        load_size=img_size,
        unique_only=unique_only,
    )

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


def _resolve_embed_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    visited: Set[int] = set()
    queue: List[Any] = [model]

    while queue:
        current = queue.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))

        embed_fn = getattr(current, "embed", None)
        if callable(embed_fn):
            return embed_fn

        for attr_name in ("base_model", "model"):
            child = getattr(current, attr_name, None)
            if child is not None:
                queue.append(child)

    raise AttributeError("Could not find an embed(...) method in the loaded model.")


def extract_embeddings(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    embed_fn = _resolve_embed_fn(model)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_device = "cuda" if use_cuda else "cpu"

    measured_batches = 0

    embeddings: List[torch.Tensor] = []
    paths: List[str] = []
    sample_ids: List[int] = []
    roles: List[str] = []

    with torch.no_grad():
        for imgs, rel_paths, ids, batch_roles in dataloader:
            if max_batches is not None and measured_batches >= max_batches:
                break

            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(amp_device, enabled=use_cuda):
                batch_embeds = embed_fn(imgs)

            if batch_embeds.ndim == 1:
                batch_embeds = batch_embeds.unsqueeze(0)

            embeddings.append(batch_embeds.detach().cpu())
            paths.extend(list(rel_paths))
            roles.extend(list(batch_roles))

            if torch.is_tensor(ids):
                sample_ids.extend(ids.tolist())
            else:
                sample_ids.extend([int(i) for i in ids])

            measured_batches += 1

    if embeddings:
        all_embeddings = torch.cat(embeddings, dim=0)
    else:
        all_embeddings = torch.empty((0,), dtype=torch.float32)

    return {
        "embeddings": all_embeddings,
        "paths": paths,
        "ids": torch.tensor(sample_ids, dtype=torch.long),
        "roles": roles,
        "samples": all_embeddings.shape[0],
        "batches": measured_batches,
    }


def run_embedding_extraction(
    dataset_root: str,
    cache_dir: str,
    output_path: str,
    split: str = "all",
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = 224,
    device: str = "cuda",
    pretrained: bool = True,
    normalize_embeds: bool = True,
    use_patch_model: bool = False,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    attention_module: str = "benchmark",
    unique_only: bool = True,
    results_csv: str = "./reports/embedding_runs.csv",
) -> Dict[str, Any]:
    dataloader = create_single_image_dataloader(
        dataset_root=dataset_root,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        unique_only=unique_only,
    )

    model, _ = dreamsim(
        pretrained=pretrained,
        device=device,
        cache_dir=cache_dir,
        normalize_embeds=normalize_embeds,
        use_patch_model=use_patch_model,
        attention_module=attention_module,
    )

    extraction = extract_embeddings(
        dataloader=dataloader,
        model=model,
        device=device,
        max_batches=max_batches,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "embeddings": extraction["embeddings"],
        "paths": extraction["paths"],
        "ids": extraction["ids"],
        "roles": extraction["roles"],
        "config": {
            "dataset_root": dataset_root,
            "cache_dir": cache_dir,
            "split": split,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "img_size": img_size,
            "device": device,
            "pretrained": pretrained,
            "normalize_embeds": normalize_embeds,
            "use_patch_model": use_patch_model,
            "attention_module": attention_module,
            "max_batches": max_batches,
            "max_samples": max_samples,
            "unique_only": unique_only,
        },
    }
    torch.save(payload, output_path)

    record = {
        "split": split,
        "attention_module": attention_module,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "img_size": img_size,
        "use_patch_model": use_patch_model,
        "normalize_embeds": normalize_embeds,
        "pretrained": pretrained,
        "max_batches": max_batches,
        "max_samples": max_samples,
        "unique_only": unique_only,
        "output_path": output_path,
        "samples": extraction["samples"],
        "batches": extraction["batches"],
    }
    log_result(results_csv, record)

    print(
        f"split={split}, attention_module={attention_module}: "
        f"saved={output_path}, "
        f"samples={extraction['samples']}"
    )

    return record


def parse_args():
    parser = argparse.ArgumentParser(description="Extract DreamSim embeddings for single NIGHTS images.")
    parser.add_argument("--dataset_root", type=str, default="./nights")
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--output_path", type=str, default="./training/embeddings/nights_embeddings.pt")
    parser.add_argument("--results_csv", type=str, default="./reports/embedding_runs.csv")
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--attention_module", type=str, default="benchmark")
    parser.add_argument("--use_patch_model", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_normalize_embeds", action="store_true")
    parser.add_argument("--keep_duplicates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    run_embedding_extraction(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        device=args.device,
        pretrained=not args.no_pretrained,
        normalize_embeds=not args.no_normalize_embeds,
        use_patch_model=args.use_patch_model,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        attention_module=args.attention_module,
        unique_only=not args.keep_duplicates,
        results_csv=args.results_csv,
    )


if __name__ == "__main__":
    main()
