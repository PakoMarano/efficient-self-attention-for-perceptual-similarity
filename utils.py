import csv
import os
from typing import Dict, Any, Optional, Tuple

import torch


def log_result(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_amp_device(device: str) -> Tuple[bool, str]:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_device = "cuda" if use_cuda else "cpu"
    return use_cuda, amp_device


def cap_max_samples(max_samples: Optional[int], dataset_len: int) -> Optional[int]:
    if max_samples is None:
        return None
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")
    return min(max_samples, dataset_len)
