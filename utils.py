import csv
import os
from typing import Dict, Any


def log_result(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = list(row.keys())

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []

        if existing_fieldnames == fieldnames:
            with open(csv_path, "a", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=fieldnames)
                writer.writerow(row)
            return

        existing_rows = list(reader)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            aligned = {key: existing_row.get(key, "") for key in fieldnames}
            writer.writerow(aligned)
        writer.writerow(row)
