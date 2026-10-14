from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Keep resumable state outside the project tree and separate from final results.
PROCESS_ROOT = ROOT.parents[1] / "codex_proc" / "adaptive_ic" / "checkpoints"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_preprocess import load_dataset
from src.baseline_model import SimulationConfig, InformationDynamicsSimulator, save_result


def run_one(
    dataset: str,
    method: str,
    output_root: Path,
    params: Dict[str, Any],
    seed: int,
    data_root: Path = ROOT / "data",
    metadata: Dict[str, Any] | None = None,
) -> Path:
    started = time.perf_counter()
    cfg_values = dict(params)
    cfg_values.update({"dataset": dataset, "method": method, "seed": int(seed)})
    config = SimulationConfig(**cfg_values)
    output = output_root / dataset / method / f"seed_{seed}"
    checkpoint_key = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:20]
    checkpoint_path = PROCESS_ROOT / checkpoint_key / "state.pkl"
    preprocessing_started = time.perf_counter()
    data = load_dataset(dataset, data_root, num_items=config.num_items, seed=seed)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    simulator = InformationDynamicsSimulator(data, config)
    result = simulator.run(checkpoint_path=checkpoint_path, checkpoint_every=100, resume=True)
    result["summary"]["preprocessing_seconds"] = preprocessing_seconds
    result["summary"]["total_seconds"] = time.perf_counter() - started
    if metadata:
        result["summary"].update(metadata)
    save_result(result, output, config, metadata=metadata)
    return output


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=["video", "news", "both"], default="both")
    p.add_argument("--users", type=int, default=None, help="override the configured user count")
    p.add_argument("--iters", type=int, default=None, help="override the configured iteration count")
    p.add_argument("--items", type=int, default=None, help="override the configured item count")
    p.add_argument("--num-rec", type=int, default=None, help="override the configured recommendation length")
    p.add_argument("--config-root", default=str(ROOT / "configs"))
    p.add_argument("--output-root", default=str(ROOT / "results" / "raw"))
    p.add_argument("--quick", action="store_true", help="small smoke test only")
    return p


def selected_datasets(value: str):
    return ["video", "news"] if value == "both" else [value]


def canonical_params(args, dataset: str, users: int, iters: int) -> Dict[str, Any]:
    config_root = Path(args.config_root)
    dataset_cfg = load_yaml(config_root / f"{dataset}_baseline.yaml")
    params = {
        "num_users": users,
        "num_items": args.items if args.items is not None else dataset_cfg.get("num_items", 10000),
        "num_iter": iters,
        "num_rec": args.num_rec if args.num_rec is not None else dataset_cfg.get("num_rec", 10),
        "alpha": dataset_cfg.get("alpha", 0.0),
        "beta": dataset_cfg.get("beta", 7.0),
        "gamma_plus": dataset_cfg.get("gamma_plus", 1.0),
        "gamma_minus": dataset_cfg.get("gamma_minus", -0.1),
        "sigma": dataset_cfg.get("sigma", 0.01),
    }
    return params


def output_root_for(args, experiment: str) -> Path:
    """Keep smoke outputs separate from formal results by default."""
    base = Path(args.output_root)
    if args.quick and base.name.lower() != "smoke":
        base = base.parent / "smoke" if base.name.lower() == "raw" else base / "smoke"
    return base / experiment


def expand_parameter_group(base: Dict[str, Any], group: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Expand one named parameter group into independent cases."""
    name = str(group["name"])
    values = group.get("values", {})
    keys = list(values)
    if not keys:
        yield name, dict(base)
        return
    choices = [values[key] if isinstance(values[key], list) else [values[key]] for key in keys]
    for combo in product(*choices):
        params = dict(base)
        params.update(dict(zip(keys, combo)))
        yield name, params


def parameter_case_name(group_name: str, params: Dict[str, Any]) -> str:
    """Return a unique, filesystem-safe label for one reproduction case."""
    if group_name == "beta":
        return f"beta_{float(params['beta']):g}"
    if group_name == "gamma_plus":
        return f"gamma_plus_{float(params['gamma_plus']):g}"
    if group_name == "gamma_minus":
        return f"gamma_minus_{float(params['gamma_minus']):g}"
    if group_name == "sigma":
        return f"sigma_{float(params['sigma']):g}"
    return group_name


def main_reproduction(argv=None):
    p = common_parser("Reproduce key settings from the paper")
    args = p.parse_args(argv)
    grid = load_yaml(Path(args.config_root) / "experiment_grid.yaml")
    reproduction = grid.get("reproduction", {})
    if args.quick:
        users, iters, datasets = 5, 5, ["video"] if args.dataset == "both" else selected_datasets(args.dataset)
    else:
        users, iters, datasets = args.users or reproduction.get("users", 1000), args.iters or reproduction.get("iterations", 5000), selected_datasets(args.dataset)
    output_root = output_root_for(args, "reproduction")
    cases = []
    if args.quick:
        for dataset in datasets:
            params = canonical_params(args, dataset, users, iters)
            params["beta"] = 7.0
            cases.append((dataset, "beta", params))
    else:
        groups = reproduction.get("parameter_sweeps", [])
        if not groups:
            groups = [{"name": "beta", "values": {"beta": reproduction.get("betas", (0.0, 4.0, 7.0))}}]
        for dataset in datasets:
            base = canonical_params(args, dataset, users, iters)
            for group in groups:
                for group_name, params in expand_parameter_group(base, group):
                    cases.append((dataset, group_name, params))
    for dataset, group_name, params in cases:
        case_name = parameter_case_name(group_name, params)
        metadata = {
            "experiment": "reproduction",
            "parameter_group": group_name,
            "parameter_case": case_name,
            "beta": params["beta"],
            "gamma_plus": params["gamma_plus"],
            "gamma_minus": params["gamma_minus"],
            "sigma": params["sigma"],
            "smoke": bool(args.quick),
        }
        print(f"running reproduction {dataset} {group_name} params={params}")
        run_one(dataset, "baseline", output_root / group_name / case_name, params, 4, metadata=metadata)
    return 0


def main_comparison(argv=None):
    p = common_parser("Compare baseline, fixed exploration and adaptive exploration")
    args = p.parse_args(argv)
    grid = load_yaml(Path(args.config_root) / "experiment_grid.yaml")
    comparison = grid.get("comparison", {})
    if args.quick:
        users, iters, seeds = 8, 8, [4]
    else:
        users, iters, seeds = args.users or comparison.get("users", 500), args.iters or comparison.get("iterations", 3000), comparison.get("seeds", [4, 5, 6])
    datasets = ["video"] if args.quick and args.dataset == "both" else selected_datasets(args.dataset)
    output_root = output_root_for(args, "comparison")
    methods = comparison.get("methods", ["baseline", "fixed", "adaptive"])
    fixed_epsilon = comparison.get("fixed_epsilon", 0.10)
    adaptive_defaults = grid.get("adaptive", {})
    window_values = adaptive_defaults.get("history_window", [100])
    target_values = adaptive_defaults.get("target_entropy", [0.75])
    default_window = 100 if 100 in window_values else window_values[0]
    default_target = 0.75 if 0.75 in target_values else target_values[0]
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                params = canonical_params(args, dataset, users, iters)
                if method == "fixed": params["epsilon"] = fixed_epsilon
                if method == "adaptive":
                    params.update({
                        "history_window": default_window if not args.quick else 100,
                        "target_entropy": default_target if not args.quick else 0.75,
                        "epsilon_min": adaptive_defaults.get("epsilon_min", 0.02),
                        "epsilon_max": adaptive_defaults.get("epsilon_max", 0.20),
                        "gain": adaptive_defaults.get("gain", 0.40),
                    })
                metadata = {"experiment": "comparison", "smoke": bool(args.quick)}
                for key in ("history_window", "target_entropy", "epsilon"):
                    if key in params:
                        metadata[key] = params[key]
                print(f"running comparison {dataset} {method} seed={seed}")
                run_one(dataset, method, output_root, params, seed, metadata=metadata)
    return 0


def main_sensitivity(argv=None):
    p = common_parser("Run adaptive exploration sensitivity experiments")
    args = p.parse_args(argv)
    grid = load_yaml(Path(args.config_root) / "experiment_grid.yaml")
    adaptive = grid.get("adaptive", {})
    if args.quick:
        users, iters, windows, targets = 8, 8, [100], [0.75]
    else:
        users = args.users or adaptive.get("users", 150)
        iters = args.iters or adaptive.get("iterations", 500)
        windows = adaptive.get("history_window", [50, 100, 200])
        targets = adaptive.get("target_entropy", [0.60, 0.75, 0.90])
    output_root = output_root_for(args, "sensitivity")
    for window in windows:
        for target in targets:
            params = canonical_params(args, "video", users, iters)
            params.update({
                "history_window": window,
                "target_entropy": target,
                "epsilon_min": adaptive.get("epsilon_min", 0.02),
                "epsilon_max": adaptive.get("epsilon_max", 0.20),
                "gain": adaptive.get("gain", 0.40),
            })
            print(f"running sensitivity W={window} target={target}")
            metadata = {
                "experiment": "sensitivity",
                "history_window": window,
                "target_entropy": target,
                "smoke": bool(args.quick),
            }
            run_one("video", "adaptive", output_root / f"W_{window}_target_{target:.2f}", params, 4, metadata=metadata)
    return 0

if __name__ == "__main__":
    raise SystemExit(main_comparison())
