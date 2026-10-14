from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Union, List
import numpy as np

DATA_FILES = {
    "video": ("phi_v1.npy", "item_popularity_v1.npy", "user_alpha_mean_v1.npy", 20),
    "news": ("phi_news.npy", "item_popularity_news.npy", "user_alpha_mean_news.npy", 14),
}

def _normalise_prob(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probability vector must be finite, non-negative and one-dimensional")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("probability vector cannot be all zero")
    return values / total


def _validate_phi(phi: np.ndarray, expected_topics: int, name: str) -> None:
    if phi.shape != (expected_topics, expected_topics):
        raise ValueError(f"{name}: unexpected phi shape {phi.shape}")
    if not np.all(np.isfinite(phi)):
        raise ValueError(f"{name}: phi contains missing or non-finite values")
    if np.any(np.all(np.isclose(phi, 0.0), axis=1)) or np.any(np.all(np.isclose(phi, 0.0), axis=0)):
        raise ValueError(f"{name}: phi contains an all-zero topic row or column")


def _validate_positive_vector(values: np.ndarray, expected_size: int, name: str) -> None:
    if values.shape != (expected_size,):
        raise ValueError(f"{name}: unexpected vector shape {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name}: vector must be finite and strictly positive")

def load_dataset(name: str, data_root: Union[str, Path], num_items: int = 10000, seed: int = 4) -> Dict[str, Any]:
    if name not in DATA_FILES:
        raise ValueError(f"unknown dataset: {name}")
    root = Path(data_root) / name
    phi_name, popularity_name, mu_name, expected_topics = DATA_FILES[name]
    phi = np.load(root / phi_name).astype(float)
    popularity = np.load(root / popularity_name).astype(float)
    mu_user = np.load(root / mu_name).astype(float)
    _validate_phi(phi, expected_topics, name)
    if popularity.shape != (expected_topics, 2):
        raise ValueError(f"{name}: unexpected popularity shape {popularity.shape}")
    if not np.all(np.isfinite(popularity)) or np.any(popularity[:, 1] < 0):
        raise ValueError(f"{name}: popularity contains missing or negative values")
    _validate_positive_vector(mu_user, expected_topics, f"{name}: user parameters")
    topic_ids = popularity[:, 0].astype(int)
    if not np.all(popularity[:, 0] == topic_ids) or not np.array_equal(topic_ids, np.arange(expected_topics)):
        raise ValueError(f"{name}: topic IDs must be consecutive")
    item_prob = _normalise_prob(popularity[:, 1])
    if int(num_items) <= 0:
        raise ValueError(f"{name}: num_items must be positive")
    rng = np.random.default_rng(seed)
    # Match the reference implementation: draw every item independently from
    # the empirical topic popularity distribution.  Do not force one item per
    # topic, otherwise the generated catalogue is no longer the same model.
    item_topics = rng.choice(topic_ids, size=int(num_items), replace=True, p=item_prob)
    topic_items = [np.flatnonzero(item_topics == j) for j in topic_ids]
    return {
        "name": name,
        "num_topics": expected_topics,
        "num_items": int(num_items),
        "phi": phi,
        "mu_user": mu_user * 15.0,
        "item_topics": item_topics,
        "topic_items": topic_items,
        "topic_item_counts": np.bincount(item_topics, minlength=expected_topics).astype(float),
        "source_files": [str(Path(name) / x) for x in (phi_name, popularity_name, mu_name)],
    }

def validate_and_write_metadata(data_root: Union[str, Path], output_path: Union[str, Path], num_items: int = 10000, seed: int = 4) -> List[Dict[str, Any]]:
    records = []
    for name in ("video", "news"):
        bundle = load_dataset(name, data_root, num_items=num_items, seed=seed)
        records.append({
            "dataset": name,
            "num_topics": bundle["num_topics"],
            "num_items": bundle["num_items"],
            "phi_shape": list(bundle["phi"].shape),
            "mu_user_shape": list(bundle["mu_user"].shape),
            "topic_item_counts": bundle["topic_item_counts"].astype(int).tolist(),
            "source_files": bundle["source_files"],
            "validation": "passed",
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "results" / "summary" / "data_metadata.json"))
    parser.add_argument("--num-items", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(validate_and_write_metadata(args.data_root, args.output, args.num_items), ensure_ascii=False, indent=2))

