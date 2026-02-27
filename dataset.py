from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASE_SPLITS = {"train", "val", "test"}
SPECIAL_SPLITS = {"test_imagenet", "test_no_imagenet"}


def _filter_df_by_split(df: pd.DataFrame, split: str, allow_all: bool = False) -> pd.DataFrame:
    valid_splits = BASE_SPLITS | SPECIAL_SPLITS
    if allow_all:
        valid_splits = valid_splits | {"all"}

    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}")

    if split in BASE_SPLITS:
        return df[df["split"] == split]
    if split == "test_imagenet":
        return df[(df["split"] == "test") & (df["is_imagenet"] == True)]
    if split == "test_no_imagenet":
        return df[(df["split"] == "test") & (df["is_imagenet"] == False)]
    return df


def _list_relative_image_paths(root_dir: str) -> List[str]:
    rel_paths: List[str] = []
    for current_root, _, files in os.walk(root_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            abs_path = os.path.join(current_root, filename)
            rel_path = os.path.relpath(abs_path, root_dir).replace("\\", "/")
            rel_paths.append(rel_path)
    rel_paths.sort()
    return rel_paths


def _flatten_triplet_records(csv_df: pd.DataFrame, root_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in csv_df.iterrows():
        base = {
            "id": int(row["id"]),
            "split": row["split"],
            "is_imagenet": bool(row["is_imagenet"]),
        }
        for role, path_col in (("ref", "ref_path"), ("left", "left_path"), ("right", "right_path")):
            rel_path = row[path_col]
            records.append(
                {
                    **base,
                    "role": role,
                    "path": rel_path,
                    "disk_path": os.path.join(root_dir, rel_path),
                }
            )
    return records


def _build_extra_image_records(
    extra_image_roots: Sequence[str],
    start_id: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    next_id = start_id
    for root_idx, extra_root in enumerate(extra_image_roots):
        rel_paths = _list_relative_image_paths(extra_root)
        root_label = os.path.basename(os.path.normpath(extra_root)) or f"root{root_idx}"
        for rel_path in rel_paths:
            records.append(
                {
                    "id": next_id,
                    "split": "extra",
                    "is_imagenet": True,
                    "role": "extra",
                    "path": f"extra/{root_idx}_{root_label}/{rel_path}",
                    "disk_path": os.path.join(extra_root, rel_path.replace("/", os.sep)),
                }
            )
            next_id += 1
    return records


def _select_distill_indices(
    images_df: pd.DataFrame,
    teacher_map: Dict[str, torch.Tensor],
    allowed_paths: Optional[Iterable[str]],
) -> List[int]:
    allowed_set = set(allowed_paths) if allowed_paths is not None else None
    if allowed_set is not None and len(allowed_set) == 0:
        return []

    indices: List[int] = []
    for idx, row in enumerate(images_df.itertuples(index=False)):
        path = row.path
        if path not in teacher_map:
            continue
        if allowed_set is not None and row.role != "extra" and path not in allowed_set:
            # Keep extra images whenever present; they are intentionally not split-gated.
            continue
        indices.append(idx)
    return indices


def _get_preprocess_fn(load_size, interpolation):
    t = transforms.Compose([
        transforms.Resize((load_size, load_size), interpolation=interpolation),
        transforms.ToTensor()
    ])
    return lambda pil_img: t(pil_img.convert("RGB"))


def load_split_paths(root_dir: str, split: str) -> Iterable[str]:
    csv_path = os.path.join(root_dir, "data.csv")
    df = pd.read_csv(csv_path)
    df = df[df["votes"] >= 6]
    df = _filter_df_by_split(df, split)

    paths = set(df["ref_path"]).union(df["left_path"]).union(df["right_path"])
    return paths


class TwoAFCDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", load_size: int = 224,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 **kwargs):
        self.root_dir = root_dir
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6] # Filter out triplets with less than 6 unanimous votes
        self.split = split
        self.load_size = load_size
        self.interpolation = interpolation
        self.preprocess_fn = _get_preprocess_fn(self.load_size, self.interpolation)

        self.csv = _filter_df_by_split(self.csv, self.split)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        id = self.csv.iloc[idx, 0]
        p = self.csv.iloc[idx, 2].astype(np.float32)
        img_ref = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 4])))
        img_left = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 5])))
        img_right = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 6])))
        return img_ref, img_left, img_right, p, id


class SingleImageDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "all", load_size: int = 224,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 extra_image_roots: Optional[Sequence[str]] = None,
                 **kwargs):
        self.root_dir = root_dir
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6]
        self.split = split
        self.load_size = load_size
        self.interpolation = interpolation
        self.preprocess_fn = _get_preprocess_fn(self.load_size, self.interpolation)

        self.csv = _filter_df_by_split(self.csv, self.split, allow_all=True)

        records = _flatten_triplet_records(self.csv, self.root_dir)

        if extra_image_roots:
            start_id = int(self.csv['id'].max()) + 1 if len(self.csv) > 0 else 1
            records.extend(_build_extra_image_records(extra_image_roots, start_id))

        images_df = pd.DataFrame(records)
        images_df = images_df.drop_duplicates(subset=['path']).reset_index(drop=True)

        self.images = images_df

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        row = self.images.iloc[idx]
        rel_path = row['path']
        img = self.preprocess_fn(Image.open(row['disk_path']))
        return img, rel_path, int(row['id']), row['role']


class DistillImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        img_size: int,
        teacher_map: Dict[str, torch.Tensor],
        allowed_paths: Optional[Iterable[str]] = None,
        extra_image_roots: Optional[Sequence[str]] = None,
    ) -> None:
        self.base = SingleImageDataset(
            root_dir=dataset_root,
            split=split,
            load_size=img_size,
            extra_image_roots=extra_image_roots,
        )
        self.teacher_map = teacher_map
        self.indices = _select_distill_indices(self.base.images, self.teacher_map, allowed_paths)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img, rel_path, sample_id, role = self.base[base_idx]
        teacher_embed = self.teacher_map[rel_path]
        return img, teacher_embed, rel_path, sample_id, role
