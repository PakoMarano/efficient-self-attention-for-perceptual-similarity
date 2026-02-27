import csv
import os
from typing import Dict, Any


def log_result(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
