from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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

    if split not in {"train", "val", "test", "test_imagenet", "test_no_imagenet"}:
        raise ValueError(f"Invalid split: {split}")

    if split == "test_imagenet":
        df = df[(df["split"] == "test") & (df["is_imagenet"] == True)]
    elif split == "test_no_imagenet":
        df = df[(df["split"] == "test") & (df["is_imagenet"] == False)]
    else:
        df = df[df["split"] == split]

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
        
        if self.split == "train" or self.split == "val" or self.split == "test":
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        else:
            raise ValueError(f'Invalid split: {split}')

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
                 unique_only: bool = True,
                 extra_image_roots: Optional[Sequence[str]] = None,
                 include_extra_images: bool = False,
                 **kwargs):
        self.root_dir = root_dir
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6]
        self.split = split
        self.load_size = load_size
        self.interpolation = interpolation
        self.preprocess_fn = _get_preprocess_fn(self.load_size, self.interpolation)

        if self.split in {"train", "val", "test"}:
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        elif split == 'all':
            pass
        else:
            raise ValueError(f'Invalid split: {split}')

        records = []
        for _, row in self.csv.iterrows():
            records.append({
                'id': int(row['id']),
                'split': row['split'],
                'is_imagenet': bool(row['is_imagenet']),
                'role': 'ref',
                'path': row['ref_path'],
                'disk_path': os.path.join(self.root_dir, row['ref_path']),
            })
            records.append({
                'id': int(row['id']),
                'split': row['split'],
                'is_imagenet': bool(row['is_imagenet']),
                'role': 'left',
                'path': row['left_path'],
                'disk_path': os.path.join(self.root_dir, row['left_path']),
            })
            records.append({
                'id': int(row['id']),
                'split': row['split'],
                'is_imagenet': bool(row['is_imagenet']),
                'role': 'right',
                'path': row['right_path'],
                'disk_path': os.path.join(self.root_dir, row['right_path']),
            })

        if include_extra_images and extra_image_roots:
            next_id = int(self.csv['id'].max()) + 1 if len(self.csv) > 0 else 1
            for root_idx, extra_root in enumerate(extra_image_roots):
                rel_paths = _list_relative_image_paths(extra_root)
                root_label = os.path.basename(os.path.normpath(extra_root)) or f"root{root_idx}"
                for rel_path in rel_paths:
                    records.append({
                        'id': next_id,
                        'split': 'extra',
                        'is_imagenet': True,
                        'role': 'extra',
                        'path': f"extra/{root_idx}_{root_label}/{rel_path}",
                        'disk_path': os.path.join(extra_root, rel_path.replace('/', os.sep)),
                    })
                    next_id += 1

        images_df = pd.DataFrame(records)
        if unique_only:
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
        unique_only: bool = True,
        allowed_paths: Optional[Iterable[str]] = None,
        extra_image_roots: Optional[Sequence[str]] = None,
        include_extra_images: bool = False,
    ) -> None:
        self.base = SingleImageDataset(
            root_dir=dataset_root,
            split=split,
            load_size=img_size,
            unique_only=unique_only,
            extra_image_roots=extra_image_roots,
            include_extra_images=include_extra_images,
        )
        self.teacher_map = teacher_map
        allowed_set = set(allowed_paths) if allowed_paths is not None else None

        indices: List[int] = []
        if allowed_set is not None and len(allowed_set) == 0:
            self.indices = []
            return
        for idx in range(len(self.base)):
            row = self.base.images.iloc[idx]
            path = row["path"]
            if path not in self.teacher_map:
                continue
            if allowed_set is not None and row["role"] != "extra" and path not in allowed_set:
                continue
            indices.append(idx)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img, rel_path, sample_id, role = self.base[base_idx]
        teacher_embed = self.teacher_map[rel_path]
        return img, teacher_embed, rel_path, sample_id, role
