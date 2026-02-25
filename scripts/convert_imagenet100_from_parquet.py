import argparse
import io
import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PIL import Image


EXT_BY_FORMAT = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
}


def _load_label_names(dataset_root: Path) -> Dict[int, str]:
    mapping_path = dataset_root / "label2text.json"
    if not mapping_path.exists():
        return {}

    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed: Dict[int, str] = {}
    for key, value in raw.items():
        try:
            parsed[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def _extract_image_bytes(cell) -> bytes:
    if isinstance(cell, dict):
        blob = cell.get("bytes")
        if blob is not None:
            return bytes(blob)
        raise ValueError("Image cell dict does not contain 'bytes'.")

    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)

    raise TypeError(f"Unsupported image cell type: {type(cell)}")


def _save_one_image(image_bytes: bytes, out_stem: Path) -> Path:
    with Image.open(io.BytesIO(image_bytes)) as img:
        image_format = (img.format or "JPEG").upper()
        ext = EXT_BY_FORMAT.get(image_format, ".jpg")
        out_path = out_stem.with_suffix(ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if image_format in EXT_BY_FORMAT:
            with out_path.open("wb") as f:
                f.write(image_bytes)
            return out_path

        rgb_img = img.convert("RGB")
        out_path = out_stem.with_suffix(".jpg")
        rgb_img.save(out_path, format="JPEG", quality=95)
        return out_path


def _convert_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    limit: Optional[int],
    class_name_mode: str,
    label_names: Dict[int, str],
) -> int:
    data_dir = dataset_root / "data"
    shard_paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not shard_paths:
        return 0

    written = 0
    for shard_index, shard_path in enumerate(shard_paths):
        df = pd.read_parquet(shard_path)
        if "image" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"Missing required columns in {shard_path.name}; expected 'image' and 'label'."
            )

        for row_index, row in df.iterrows():
            if limit is not None and written >= limit:
                return written

            label = int(row["label"])
            if class_name_mode == "id":
                class_dir_name = f"{label:03d}"
            else:
                label_text = label_names.get(label, f"class_{label:03d}")
                class_dir_name = f"{label:03d}_{_sanitize_name(label_text)}"

            out_dir = output_root / split / class_dir_name
            out_stem = out_dir / f"{split}_{shard_index:03d}_{int(row_index):06d}"
            image_bytes = _extract_image_bytes(row["image"])
            _save_one_image(image_bytes, out_stem)
            written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local Hugging Face ImageNet100 parquet shards into image folders."
    )
    parser.add_argument("--dataset_root", type=str, default="./imagenet-100")
    parser.add_argument("--output_root", type=str, default="./imagenet-100-images")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--limit_per_split", type=int, default=None)
    parser.add_argument(
        "--class_name_mode",
        type=str,
        default="text",
        choices=["id", "text"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit_per_split is not None and args.limit_per_split <= 0:
        raise ValueError("limit_per_split must be > 0 when provided.")

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    label_names = _load_label_names(dataset_root)

    total = 0
    for split in args.splits:
        count = _convert_split(
            dataset_root=dataset_root,
            output_root=output_root,
            split=split,
            limit=args.limit_per_split,
            class_name_mode=args.class_name_mode,
            label_names=label_names,
        )
        total += count
        print(f"split={split}: wrote {count} images")

    print(f"done: wrote {total} images to {os.path.abspath(output_root)}")


if __name__ == "__main__":
    main()
