from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _experiment_from_path(path: Path) -> str:
    for name in ("comparison", "reproduction", "sensitivity"):
        if name in path.parts:
            return name
    return "unknown"


def _read_with_context(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    config_path = path.parent / "config.json"
    context = {}
    if config_path.exists():
        try:
            context = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            context = {}
    for key, value in context.items():
        if key not in frame.columns:
            frame[key] = value
    smoke = context.get("smoke")
    if smoke is None:
        # Older quick runs predated the explicit metadata flag.  Their small
        # user/iteration counts identify them without deleting any files.
        try:
            smoke = int(context.get("num_users", 0)) <= 10 and int(context.get("num_iter", 0)) <= 10
        except (TypeError, ValueError):
            smoke = False
    frame["smoke"] = bool(smoke)
    frame["result_dir"] = str(path.parent)
    if "experiment" not in frame.columns or frame["experiment"].isna().all():
        frame["experiment"] = _experiment_from_path(path)
    if "parameter_group" not in frame.columns:
        match = re.search(r"(?:reproduction|sensitivity)[\\/]([^\\/]+)", str(path.parent))
        frame["parameter_group"] = match.group(1) if match else ""
    return frame


def _parameter_setting(row: pd.Series) -> str:
    group = str(row.get("parameter_group", ""))
    if group == "beta" and pd.notna(row.get("beta")):
        return f"beta={float(row['beta']):g}"
    if group == "gamma_plus" and pd.notna(row.get("gamma_plus")):
        return f"gamma+={float(row['gamma_plus']):g}"
    if group == "gamma_minus" and pd.notna(row.get("gamma_minus")):
        return f"gamma-={float(row['gamma_minus']):g}"
    if group == "sigma" and pd.notna(row.get("sigma")):
        return f"sigma={float(row['sigma']):g}"
    if group.startswith("W_"):
        return group
    return group or "unknown"


def _plot_comparison_bars(summary: pd.DataFrame, figure_root: Path) -> List[str]:
    outputs = []
    if summary.empty:
        return outputs
    for metric, filename, ylabel in [
        ("mean_final_relative_entropy", "final_relative_entropy.png", "Final relative entropy"),
        ("mean_positive_feedback_rate", "positive_feedback_rate.png", "Simulated positive feedback rate"),
        ("mean_topic_coverage", "topic_coverage.png", "Topic coverage"),
        ("simulation_seconds", "runtime.png", "Simulation seconds"),
    ]:
        if metric not in summary:
            continue
        plt.figure(figsize=(8, 5))
        sns.barplot(data=summary, x="method", y=metric, hue="dataset", errorbar="sd")
        plt.ylabel(ylabel)
        plt.xlabel("Method")
        plt.tight_layout()
        out = figure_root / filename
        plt.savefig(out, dpi=160)
        plt.close()
        outputs.append(str(out))
    if {"mean_positive_feedback_rate", "mean_final_recommendation_entropy"}.issubset(summary.columns):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=summary,
            x="mean_final_recommendation_entropy",
            y="mean_positive_feedback_rate",
            hue="method",
            style="dataset",
            s=80,
        )
        plt.xlabel("Final recommendation entropy")
        plt.ylabel("Simulated positive feedback rate")
        plt.tight_layout()
        out = figure_root / "feedback_vs_diversity.png"
        plt.savefig(out, dpi=160)
        plt.close()
        outputs.append(str(out))
    return outputs


def _plot_reproduction(summary: pd.DataFrame, figure_root: Path) -> List[str]:
    if summary.empty:
        return []
    data = summary.copy()
    data["parameter_setting"] = data.apply(_parameter_setting, axis=1)
    outputs = []
    for metric, filename, ylabel in [
        ("mean_final_relative_entropy", "reproduction_relative_entropy.png", "Final relative entropy"),
        ("mean_positive_feedback_rate", "reproduction_positive_feedback_rate.png", "Simulated positive feedback rate"),
    ]:
        if metric not in data:
            continue
        plt.figure(figsize=(10, 5))
        sns.barplot(data=data, x="parameter_setting", y=metric, hue="dataset", errorbar=None)
        plt.ylabel(ylabel)
        plt.xlabel("Paper parameter setting")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = figure_root / filename
        plt.savefig(out, dpi=160)
        plt.close()
        outputs.append(str(out))
    return outputs


def _plot_step_metrics(steps: pd.DataFrame, figure_root: Path) -> List[str]:
    outputs = []
    if steps.empty:
        return outputs
    for metric, filename, ylabel in [
        ("mean_cumulative_entropy", "entropy_over_time.png", "Mean cumulative normalized entropy"),
        ("mean_positive_feedback_rate", "feedback_over_time.png", "Mean positive feedback rate"),
    ]:
        if metric not in steps:
            continue
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=steps, x="step", y=metric, hue="method", style="dataset", estimator="mean", errorbar="sd")
        plt.ylabel(ylabel)
        plt.tight_layout()
        out = figure_root / filename
        plt.savefig(out, dpi=160)
        plt.close()
        outputs.append(str(out))
    return outputs


def _plot_final_boxplot(finals: pd.DataFrame, figure_root: Path) -> List[str]:
    if finals.empty or "recommendation_entropy" not in finals:
        return []
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=finals, x="method", y="recommendation_entropy", hue="dataset")
    plt.tight_layout()
    out = figure_root / "final_entropy_boxplot.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return [str(out)]


def _plot_sensitivity(summary: pd.DataFrame, figure_root: Path) -> List[str]:
    if summary.empty:
        return []
    data = summary.copy()
    if "history_window" not in data or "target_entropy" not in data:
        return []
    data["history_window"] = pd.to_numeric(data["history_window"], errors="coerce")
    data["target_entropy"] = pd.to_numeric(data["target_entropy"], errors="coerce")
    data = data.dropna(subset=["history_window", "target_entropy"])
    if data.empty:
        return []
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=data,
        x="target_entropy",
        y="mean_final_recommendation_entropy",
        hue="history_window",
        marker="o",
        errorbar=None,
    )
    plt.xlabel("Target entropy")
    plt.ylabel("Mean final recommendation entropy")
    plt.tight_layout()
    out = figure_root / "sensitivity.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return [str(out)]


def plot_results(result_root: Union[str, Path], figure_root: Union[str, Path], include_smoke: bool = False) -> List[str]:
    result_root, figure_root = Path(result_root), Path(figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    (figure_root.parent / "summary").mkdir(parents=True, exist_ok=True)
    summary_paths = list(result_root.rglob("metrics_summary.csv"))
    step_paths = list(result_root.rglob("metrics_per_step.csv"))
    final_paths = list(result_root.rglob("user_final_metrics.csv"))
    outputs: List[str] = []

    summary = pd.concat([_read_with_context(p) for p in summary_paths], ignore_index=True) if summary_paths else pd.DataFrame()
    if not summary.empty:
        if not include_smoke and "smoke" in summary:
            summary = summary[~summary["smoke"].fillna(False)].copy()
        summary.to_csv(figure_root.parent / "summary" / "all_summaries.csv", index=False)
        outputs.extend(_plot_comparison_bars(summary[summary["experiment"] == "comparison"], figure_root))
        outputs.extend(_plot_reproduction(summary[summary["experiment"] == "reproduction"], figure_root))
        outputs.extend(_plot_sensitivity(summary[summary["experiment"] == "sensitivity"], figure_root))

    if step_paths:
        steps = pd.concat([_read_with_context(p) for p in step_paths], ignore_index=True)
        if not include_smoke and "smoke" in steps:
            steps = steps[~steps["smoke"].fillna(False)].copy()
        steps["dataset"] = steps["result_dir"].str.extract(r"(video|news)", expand=False)
        steps["method"] = steps["result_dir"].str.extract(r"(baseline|fixed|adaptive)", expand=False)
        outputs.extend(_plot_step_metrics(steps[steps["experiment"] == "comparison"], figure_root))

    if final_paths:
        finals = pd.concat([_read_with_context(p) for p in final_paths], ignore_index=True)
        if not include_smoke and "smoke" in finals:
            finals = finals[~finals["smoke"].fillna(False)].copy()
        finals["dataset"] = finals["result_dir"].str.extract(r"(video|news)", expand=False)
        finals["method"] = finals["result_dir"].str.extract(r"(baseline|fixed|adaptive)", expand=False)
        outputs.extend(_plot_final_boxplot(finals[finals["experiment"] == "comparison"], figure_root))
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--figure-root", default="results/figures")
    parser.add_argument("--include-smoke", action="store_true", help="include smoke-test outputs in figures")
    args = parser.parse_args()
    print("\n".join(plot_results(args.result_root, args.figure_root, include_smoke=args.include_smoke)))
