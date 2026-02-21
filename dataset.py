from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os


def _get_preprocess_fn(load_size, interpolation):
    t = transforms.Compose([
        transforms.Resize((load_size, load_size), interpolation=interpolation),
        transforms.ToTensor()
    ])
    return lambda pil_img: t(pil_img.convert("RGB"))


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
                 unique_only: bool = True, **kwargs):
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
            })
            records.append({
                'id': int(row['id']),
                'split': row['split'],
                'is_imagenet': bool(row['is_imagenet']),
                'role': 'left',
                'path': row['left_path'],
            })
            records.append({
                'id': int(row['id']),
                'split': row['split'],
                'is_imagenet': bool(row['is_imagenet']),
                'role': 'right',
                'path': row['right_path'],
            })

        images_df = pd.DataFrame(records)
        if unique_only:
            images_df = images_df.drop_duplicates(subset=['path']).reset_index(drop=True)

        self.images = images_df

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        row = self.images.iloc[idx]
        rel_path = row['path']
        img = self.preprocess_fn(Image.open(os.path.join(self.root_dir, rel_path)))
        return img, rel_path, int(row['id']), row['role']
