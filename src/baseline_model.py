from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import json
import pickle
import time
import numpy as np
import pandas as pd
import yaml


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    total = weights.sum()
    return weights / total if total > 0 else np.full_like(weights, 1.0 / len(weights))


def entropy_rows(counts: np.ndarray, normalised: bool = False) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    safe_probs = np.where(probs > 0, probs, 1.0)
    terms = np.where(probs > 0, probs * np.log(safe_probs), 0.0)
    result = -terms.sum(axis=1)
    if normalised and counts.shape[1] > 1:
        result = result / np.log(counts.shape[1])
    return result

@dataclass
class SimulationConfig:
    dataset: str
    method: str = "baseline"
    num_users: int = 1000
    num_items: int = 10000
    num_iter: int = 5000
    num_rec: int = 10
    dt: float = 0.01
    alpha: float = 0.0
    beta: float = 7.0
    gamma_plus: float = 1.0
    gamma_minus: float = -0.1
    sigma: float = 0.01
    epsilon: float = 0.10
    history_window: int = 100
    target_entropy: float = 0.75
    epsilon_min: float = 0.02
    epsilon_max: float = 0.20
    gain: float = 0.40
    seed: int = 4

    def __post_init__(self) -> None:
        if self.method not in {"baseline", "fixed", "adaptive"}:
            raise ValueError(f"unknown method: {self.method}")
        for name in ("num_users", "num_items", "num_iter", "num_rec"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_rec > self.num_items:
            raise ValueError("num_rec cannot exceed num_items when recommendations must be unique")
        if self.alpha != 0.0:
            raise ValueError("alpha is not part of this topic-level implementation; only alpha=0 is supported")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError("dt must be a positive finite number")
        if not 0.0 <= self.epsilon_min <= self.epsilon_max <= 1.0:
            raise ValueError("epsilon_min and epsilon_max must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.target_entropy <= 1.0:
            raise ValueError("target_entropy must be in [0, 1]")
        if self.history_window <= 0 or self.gain < 0:
            raise ValueError("history_window must be positive and gain must be non-negative")

class InformationDynamicsSimulator:
    """Fast topic-equivalent implementation of the author's item-level model."""
    def __init__(self, data: Dict[str, Any], config: SimulationConfig):
        self.data = data
        self.config = config
        if int(data["num_items"]) < config.num_rec:
            raise ValueError("data does not contain enough items for a unique recommendation list")
        self.rng = np.random.default_rng(config.seed)
        self.n = int(config.num_users)
        self.t = int(data["num_topics"])
        self.num_rec = int(config.num_rec)
        self.obs = self.rng.dirichlet(data["mu_user"], size=self.n)
        self.intrinsic = self.rng.dirichlet(data["mu_user"], size=self.n)
        self.item_topics = data["item_topics"]
        self.topic_items = data["topic_items"]
        self.topic_counts = data["topic_item_counts"].astype(float)
        self.phi = data["phi"]
        self.history_window = max(1, int(config.history_window))
        self.history = np.full((self.n, self.history_window), -1, dtype=np.int16)
        self.history_counts = np.zeros((self.n, self.t), dtype=np.int32)
        self.history_len = np.zeros(self.n, dtype=np.int32)
        self.history_pos = np.zeros(self.n, dtype=np.int32)
        self.cumulative_rec = np.zeros((self.n, self.t), dtype=np.float64)
        self.cumulative_pos = np.zeros(self.n, dtype=np.int64)
        self.cumulative_total = np.zeros(self.n, dtype=np.int64)
        self.rows = []

    def _entropy_control(self) -> np.ndarray:
        h = entropy_rows(self.history_counts, normalised=True)
        cfg = self.config
        if cfg.method != "adaptive":
            return np.full(self.n, cfg.epsilon if cfg.method == "fixed" else 0.0)
        epsilon = cfg.epsilon_min + cfg.gain * (cfg.target_entropy - h)
        epsilon = np.clip(epsilon, cfg.epsilon_min, cfg.epsilon_max)
        epsilon[self.history_len == 0] = cfg.epsilon_min
        return epsilon

    def _add_history(self, rec_topics: np.ndarray) -> None:
        for i in range(self.n):
            for topic in rec_topics[i]:
                pos = self.history_pos[i]
                old = self.history[i, pos]
                if old >= 0:
                    self.history_counts[i, old] -= 1
                else:
                    self.history_len[i] = min(self.history_len[i] + 1, self.history_window)
                self.history[i, pos] = int(topic)
                self.history_counts[i, int(topic)] += 1
                self.history_pos[i] = (pos + 1) % self.history_window

    def _sample_for_user(self, user_index: int, topic_scores: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chosen_topics = np.empty(self.num_rec, dtype=np.int32)
        chosen_items = np.empty(self.num_rec, dtype=np.int64)
        explored = np.zeros(self.num_rec, dtype=bool)
        used_by_topic = [set() for _ in range(self.t)]
        remaining = self.topic_counts.copy()
        base_logits = self.config.beta * topic_scores
        for slot in range(self.num_rec):
            available = np.flatnonzero(remaining > 0)
            if available.size == 0:
                raise RuntimeError("no items remain before the recommendation list is complete")
            do_explore = bool(self.rng.random() < epsilon)
            if do_explore:
                topic = int(self.rng.choice(available))
                explored[slot] = True
            else:
                logits = base_logits[available] + np.log(np.maximum(remaining[available], 1.0))
                probs = stable_softmax(logits)
                topic = int(self.rng.choice(available, p=probs))
            pool = self.topic_items[topic]
            topic_used = used_by_topic[topic]
            # Most topic pools are much larger than a recommendation list.  A
            # rejection draw avoids scanning the whole pool on every slot;
            # retain the exact free-pool fallback as a safe path near exhaustion.
            if remaining[topic] <= max(3.0, len(pool) / 2.0):
                free = np.asarray([x for x in pool if int(x) not in topic_used], dtype=np.int64)
                if free.size == 0:
                    raise RuntimeError("topic sampling selected an exhausted item pool")
                item = int(self.rng.choice(free))
            else:
                while True:
                    item = int(pool[self.rng.integers(len(pool))])
                    if item not in topic_used:
                        break
            topic_used.add(item)
            remaining[topic] = max(0.0, remaining[topic] - 1.0)
            chosen_topics[slot] = topic
            chosen_items[slot] = item
        return chosen_topics, explored, chosen_items

    def _recommend(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        topic_scores = self.obs @ self.phi
        epsilons = self._entropy_control()
        rec_topics = np.empty((self.n, self.num_rec), dtype=np.int32)
        explored = np.zeros((self.n, self.num_rec), dtype=bool)
        rec_items = np.empty((self.n, self.num_rec), dtype=np.int64)
        for i in range(self.n):
            rec_topics[i], explored[i], rec_items[i] = self._sample_for_user(i, topic_scores[i], float(epsilons[i]))
        return rec_topics, explored, epsilons, rec_items

    def _update_feedback(self, rec_topics: np.ndarray) -> np.ndarray:
        rows = np.arange(self.n)[:, None]
        like_prob = self.intrinsic[rows, rec_topics]
        feedback = self.rng.random(like_prob.shape) < like_prob
        pos_counts = np.zeros((self.n, self.t), dtype=float)
        neg_counts = np.zeros((self.n, self.t), dtype=float)
        for topic in range(self.t):
            pos_counts[:, topic] = np.sum((rec_topics == topic) & feedback, axis=1)
            neg_counts[:, topic] = np.sum((rec_topics == topic) & (~feedback), axis=1)
        pos_total = pos_counts.sum(axis=1, keepdims=True)
        neg_total = neg_counts.sum(axis=1, keepdims=True)
        delta_pos = self.config.gamma_plus * (pos_counts - pos_total * self.obs) * self.config.dt
        delta_neg = self.config.gamma_minus * (neg_counts - neg_total * self.obs) * self.config.dt
        noise = self.config.sigma * self.rng.normal(0.0, np.sqrt(self.config.dt), size=self.obs.shape)
        self.obs = np.clip(self.obs + delta_pos + delta_neg + noise, 0.0, 1.0)
        sums = self.obs.sum(axis=1, keepdims=True)
        zero = sums[:, 0] <= 0
        self.obs = np.divide(self.obs, sums, out=np.full_like(self.obs, 1.0 / self.t), where=sums > 0)
        if np.any(zero):
            self.obs[zero] = 1.0 / self.t
        return feedback

    def _step_metrics(self, rec_topics: np.ndarray, feedback: np.ndarray, epsilons: np.ndarray, explored: np.ndarray, step: int) -> dict:
        current_counts = np.zeros((self.n, self.t), dtype=float)
        for topic in range(self.t):
            current_counts[:, topic] = np.sum(rec_topics == topic, axis=1)
        self.cumulative_rec += current_counts
        self.cumulative_pos += feedback.sum(axis=1)
        self.cumulative_total += feedback.shape[1]
        entropy_now = entropy_rows(current_counts, normalised=True)
        cumulative_entropy = entropy_rows(self.cumulative_rec, normalised=True)
        rec_prob = self.cumulative_rec / np.maximum(self.cumulative_rec.sum(axis=1, keepdims=True), 1.0)
        intrinsic_entropy = entropy_rows(self.intrinsic, normalised=True)
        over_mass, under_mass, over_degree, under_degree = self._recommendation_bias_metrics(rec_prob)
        return {
            "step": step,
            "method": self.config.method,
            "mean_recommendation_entropy": float(np.mean(entropy_now)),
            "std_recommendation_entropy": float(np.std(entropy_now)),
            "mean_cumulative_entropy": float(np.mean(cumulative_entropy)),
            "std_cumulative_entropy": float(np.std(cumulative_entropy)),
            "mean_relative_entropy": float(np.mean(cumulative_entropy / np.maximum(intrinsic_entropy, 1e-12))),
            "mean_positive_feedback_rate": float(np.mean(feedback)),
            "mean_negative_feedback_rate": float(np.mean(~feedback)),
            "positive_feedback_count": int(feedback.sum()),
            "negative_feedback_count": int((~feedback).sum()),
            "feedback_count": int(feedback.size),
            "mean_topic_coverage": float(np.mean(np.sum(current_counts > 0, axis=1) / self.t)),
            "mean_cumulative_topic_coverage": float(np.mean(np.sum(self.cumulative_rec > 0, axis=1) / self.t)),
            "mean_over_recommendation_mass": float(np.mean(over_mass)),
            "mean_under_recommendation_mass": float(np.mean(under_mass)),
            "mean_over_recommendation_degree": float(np.mean(over_degree)),
            "mean_under_recommendation_degree": float(np.mean(under_degree)),
            "mean_epsilon": float(np.mean(epsilons)),
            "exploration_fraction": float(np.mean(explored)),
        }

    def _final_user_metrics(self) -> pd.DataFrame:
        intrinsic_entropy = entropy_rows(self.intrinsic, normalised=True)
        recommendation_entropy = entropy_rows(self.cumulative_rec, normalised=True)
        rec_prob = self.cumulative_rec / np.maximum(self.cumulative_rec.sum(axis=1, keepdims=True), 1.0)
        over_mass, under_mass, over_degree, under_degree = self._recommendation_bias_metrics(rec_prob)
        return pd.DataFrame({
            "user_id": np.arange(self.n),
            "intrinsic_entropy": intrinsic_entropy,
            "recommendation_entropy": recommendation_entropy,
            "relative_entropy": recommendation_entropy / np.maximum(intrinsic_entropy, 1e-12),
            "topic_coverage": np.sum(self.cumulative_rec > 0, axis=1) / self.t,
            "positive_feedback_rate": self.cumulative_pos / np.maximum(self.cumulative_total, 1),
            "negative_feedback_rate": 1.0 - self.cumulative_pos / np.maximum(self.cumulative_total, 1),
            "over_recommendation_mass": over_mass,
            "under_recommendation_mass": under_mass,
            "over_recommendation_degree": over_degree,
            "under_recommendation_degree": under_degree,
        })

    def _recommendation_bias_metrics(self, rec_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match the reference implementation's recommendation bias metrics.

        The reference rounds recommendation minus intrinsic interest to five
        decimals, measures both over- and under-recommendation mass on the
        recommendation distribution, and normalises each degree by the
        intrinsic-interest mass in the affected topics.  Under-recommendation
        degree therefore remains signed (negative), as in the paper code.
        """
        diff = np.around(np.asarray(rec_prob, dtype=float) - self.intrinsic, 5)
        over_mask = diff > 0
        under_mask = diff < 0
        over_mass = np.sum(rec_prob * over_mask, axis=1)
        under_mass = np.sum(rec_prob * under_mask, axis=1)
        over_numerator = np.sum(diff * over_mask, axis=1)
        under_numerator = np.sum(diff * under_mask, axis=1)
        over_denominator = np.sum(self.intrinsic * over_mask, axis=1)
        under_denominator = np.sum(self.intrinsic * under_mask, axis=1)
        over_degree = np.divide(
            over_numerator,
            over_denominator,
            out=np.zeros_like(over_numerator),
            where=over_denominator > 0,
        )
        under_degree = np.divide(
            under_numerator,
            under_denominator,
            out=np.zeros_like(under_numerator),
            where=under_denominator > 0,
        )
        return over_mass, under_mass, over_degree, under_degree

    def _checkpoint_payload(self, next_step: int, simulation_seconds: float) -> Dict[str, Any]:
        return {
            "version": 1,
            "config": dict(self.config.__dict__),
            "next_step": int(next_step),
            "simulation_seconds": float(simulation_seconds),
            "obs": self.obs,
            "intrinsic": self.intrinsic,
            "history": self.history,
            "history_counts": self.history_counts,
            "history_len": self.history_len,
            "history_pos": self.history_pos,
            "cumulative_rec": self.cumulative_rec,
            "cumulative_pos": self.cumulative_pos,
            "cumulative_total": self.cumulative_total,
            "rows": self.rows,
            "rng_state": self.rng.bit_generator.state,
        }

    def _write_checkpoint(self, checkpoint_path: Union[str, Path], next_step: int, simulation_seconds: float) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self._checkpoint_payload(next_step, simulation_seconds), protocol=pickle.HIGHEST_PROTOCOL))

    def _restore_checkpoint(self, checkpoint_path: Union[str, Path]) -> Tuple[int, float]:
        payload = pickle.loads(Path(checkpoint_path).read_bytes())
        if payload.get("version") != 1 or payload.get("config") != dict(self.config.__dict__):
            return 0, 0.0
        self.obs = np.asarray(payload["obs"])
        self.intrinsic = np.asarray(payload["intrinsic"])
        self.history = np.asarray(payload["history"])
        self.history_counts = np.asarray(payload["history_counts"])
        self.history_len = np.asarray(payload["history_len"])
        self.history_pos = np.asarray(payload["history_pos"])
        self.cumulative_rec = np.asarray(payload["cumulative_rec"])
        self.cumulative_pos = np.asarray(payload["cumulative_pos"])
        self.cumulative_total = np.asarray(payload["cumulative_total"])
        self.rows = list(payload["rows"])
        self.rng.bit_generator.state = payload["rng_state"]
        return int(payload["next_step"]), float(payload.get("simulation_seconds", 0.0))

    def run(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        checkpoint_every: int = 100,
        resume: bool = True,
    ) -> Dict[str, Any]:
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        started = time.perf_counter()
        start_step = 0
        previous_simulation_seconds = 0.0
        if checkpoint_path is not None and resume and Path(checkpoint_path).exists():
            try:
                start_step, previous_simulation_seconds = self._restore_checkpoint(checkpoint_path)
            except (OSError, EOFError, pickle.PickleError, KeyError, ValueError, TypeError):
                start_step, previous_simulation_seconds = 0, 0.0
        simulation_started = time.perf_counter()
        for step in range(start_step, self.config.num_iter):
            rec_topics, explored, epsilons, _rec_items = self._recommend()
            feedback = self._update_feedback(rec_topics)
            self._add_history(rec_topics)
            self.rows.append(self._step_metrics(rec_topics, feedback, epsilons, explored, step))
            completed = step + 1
            if checkpoint_path is not None and (completed % checkpoint_every == 0 or completed == self.config.num_iter):
                elapsed = previous_simulation_seconds + time.perf_counter() - simulation_started
                self._write_checkpoint(checkpoint_path, completed, elapsed)
        simulation_seconds = previous_simulation_seconds + time.perf_counter() - simulation_started
        metrics_started = time.perf_counter()
        per_step = pd.DataFrame(self.rows)
        final = self._final_user_metrics()
        metrics_seconds = time.perf_counter() - metrics_started
        summary = {
            "dataset": self.config.dataset,
            "method": self.config.method,
            "seed": self.config.seed,
            "num_users": self.n,
            "num_iter": self.config.num_iter,
            "num_topics": self.t,
            "mean_final_intrinsic_entropy": float(final["intrinsic_entropy"].mean()),
            "std_final_intrinsic_entropy": float(final["intrinsic_entropy"].std(ddof=0)),
            "mean_final_relative_entropy": float(final["relative_entropy"].mean()),
            "std_final_relative_entropy": float(final["relative_entropy"].std(ddof=0)),
            "mean_final_recommendation_entropy": float(final["recommendation_entropy"].mean()),
            "std_final_recommendation_entropy": float(final["recommendation_entropy"].std(ddof=0)),
            "mean_positive_feedback_rate": float(final["positive_feedback_rate"].mean()),
            "mean_negative_feedback_rate": float(final["negative_feedback_rate"].mean()),
            "mean_topic_coverage": float(final["topic_coverage"].mean()),
            "mean_over_recommendation_mass": float(final["over_recommendation_mass"].mean()),
            "mean_under_recommendation_mass": float(final["under_recommendation_mass"].mean()),
            "mean_over_recommendation_degree": float(final["over_recommendation_degree"].mean()),
            "mean_under_recommendation_degree": float(final["under_recommendation_degree"].mean()),
            "mean_epsilon": float(per_step["mean_epsilon"].mean()),
            "simulation_seconds": simulation_seconds,
            "metrics_seconds": metrics_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        return {"per_step": per_step, "final": final, "summary": summary}

def save_result(
    result: Dict[str, Any],
    output_dir: Union[str, Path],
    config: SimulationConfig,
    metadata: Dict[str, Any] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_payload = dict(config.__dict__)
    if metadata:
        config_payload.update(metadata)
    result["per_step"].to_csv(output / "metrics_per_step.csv", index=False)
    result["final"].to_csv(output / "user_final_metrics.csv", index=False)
    pd.DataFrame([result["summary"]]).to_csv(output / "metrics_summary.csv", index=False)
    (output / "seed.txt").write_text(str(config.seed), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    runtime = {
        key: result["summary"][key]
        for key in ("preprocessing_seconds", "simulation_seconds", "metrics_seconds", "total_seconds")
        if key in result["summary"]
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")



