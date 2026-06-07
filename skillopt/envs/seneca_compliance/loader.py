from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def _normalize_item(raw: dict) -> dict:
    return {
        "id": str(raw.get("id") or ""),
        "question": str(raw.get("task") or ""),
        "ground_truth": str(raw.get("ground_truth") or ""),
        "task_type": str(raw.get("role") or "markdown-compliance"),
        "broken_md": str(raw.get("context", {}).get("broken_md") or ""),
        "rules_ref": raw.get("context", {}).get("rules_ref") or [],
        "grading": raw.get("grading") or {"type": "validator_exit_code", "expect_exit": 0}
    }


class SenecaComplianceLoader(SplitDataLoader):
    def load_split_items(self, split_path: str) -> list[dict]:
        path = Path(split_path)
        json_files = sorted(path.glob("*.json"))
        if json_files:
            with json_files[0].open(encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(
                    f"Expected JSON array at top level of {json_files[0]}"
                )
            return [_normalize_item(row) for row in payload]
        raise FileNotFoundError(
            f"No .json file found in {split_path}"
        )
