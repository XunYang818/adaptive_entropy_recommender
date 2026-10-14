from __future__ import annotations
from pathlib import Path
from typing import Union, Dict, Any
import pandas as pd
import numpy as np

def summarise_result(result_dir: Union[str, Path]) -> Dict[str, Any]:
    result_dir = Path(result_dir)
    summary = pd.read_csv(result_dir / "metrics_summary.csv").iloc[0].to_dict()
    return summary

def combine_summaries(result_root: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
    result_root = Path(result_root)
    rows = []
    for path in result_root.rglob("metrics_summary.csv"):
        try:
            rows.append(pd.read_csv(path).iloc[0].to_dict())
        except (IndexError, pd.errors.EmptyDataError):
            continue
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(output_path, index=False)
    return frame

def paired_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    numeric = [c for c in [
        "mean_final_relative_entropy", "std_final_relative_entropy",
        "mean_final_recommendation_entropy", "std_final_recommendation_entropy",
        "mean_positive_feedback_rate", "mean_negative_feedback_rate",
        "mean_topic_coverage", "mean_over_recommendation_mass",
        "mean_under_recommendation_mass", "mean_over_recommendation_degree",
        "mean_under_recommendation_degree", "simulation_seconds", "metrics_seconds",
        "total_seconds",
    ] if c in frame]
    keys = [c for c in ["dataset", "method"] if c in frame]
    return frame.groupby(keys, as_index=False)[numeric].agg(["mean", "std"]).reset_index()
