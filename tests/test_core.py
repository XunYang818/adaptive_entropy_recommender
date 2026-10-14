import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_preprocess import load_dataset
from src.baseline_model import SimulationConfig, InformationDynamicsSimulator, stable_softmax

class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset("video", ROOT / "data", num_items=500, seed=4)

    def test_softmax(self):
        self.assertAlmostEqual(float(stable_softmax(np.array([1., 2., 3.])).sum()), 1.0, places=10)

    def test_simulation_invariants(self):
        cfg = SimulationConfig(dataset="video", method="adaptive", num_users=6, num_items=500, num_iter=4,
                               history_window=3, target_entropy=0.75, seed=4)
        sim = InformationDynamicsSimulator(self.data, cfg)
        result = sim.run()
        self.assertEqual(len(result["per_step"]), 4)
        self.assertTrue(np.all(sim.obs >= 0))
        self.assertTrue(np.allclose(sim.obs.sum(axis=1), 1.0))
        self.assertTrue(((result["per_step"]["mean_epsilon"] >= cfg.epsilon_min) &
                         (result["per_step"]["mean_epsilon"] <= cfg.epsilon_max)).all())
        self.assertTrue(((result["final"]["recommendation_entropy"] >= 0) &
                         (result["final"]["recommendation_entropy"] <= 1)).all())
        self.assertTrue(np.allclose(result["per_step"]["positive_feedback_count"] +
                                    result["per_step"]["negative_feedback_count"],
                                    result["per_step"]["feedback_count"]))
        self.assertTrue(((result["per_step"]["mean_negative_feedback_rate"] >= 0) &
                         (result["per_step"]["mean_negative_feedback_rate"] <= 1)).all())
        self.assertIn("mean_final_recommendation_entropy", result["summary"])
        self.assertIn("std_final_recommendation_entropy", result["summary"])

    def test_baseline_and_fixed(self):
        for method in ("baseline", "fixed"):
            cfg = SimulationConfig(dataset="video", method=method, num_users=4, num_items=500, num_iter=2, seed=7)
            result = InformationDynamicsSimulator(self.data, cfg).run()
            self.assertEqual(len(result["final"]), 4)

    def test_controller_does_not_read_intrinsic_interest(self):
        cfg = SimulationConfig(dataset="video", method="adaptive", num_users=3, num_items=500, num_iter=1, seed=9)
        first = InformationDynamicsSimulator(self.data, cfg)
        second = InformationDynamicsSimulator(self.data, cfg)
        second.intrinsic[:] = np.flip(second.intrinsic, axis=1)
        a = first._recommend()
        b = second._recommend()
        self.assertTrue(np.array_equal(a[0], b[0]))
        self.assertTrue(np.array_equal(a[1], b[1]))
        self.assertTrue(np.allclose(a[2], b[2]))

    def test_recommendation_items_are_unique(self):
        cfg = SimulationConfig(dataset="video", method="adaptive", num_users=6, num_items=500, num_iter=1, seed=4)
        sim = InformationDynamicsSimulator(self.data, cfg)
        _topics, _explored, _epsilons, items = sim._recommend()
        for row in items:
            self.assertEqual(len(np.unique(row)), cfg.num_rec)

    def test_beta_trend_increases_concentration(self):
        item_topics = np.array([0] * 100 + [1] * 100, dtype=np.int64)
        toy_data = {
            "num_topics": 2,
            "num_items": 200,
            "phi": np.eye(2),
            "mu_user": np.ones(2) * 3.0,
            "item_topics": item_topics,
            "topic_items": [np.flatnonzero(item_topics == j) for j in range(2)],
            "topic_item_counts": np.array([100.0, 100.0]),
        }
        entropies = []
        for beta in (0.0, 7.0):
            cfg = SimulationConfig(
                dataset="toy", method="baseline", num_users=64, num_items=200,
                num_iter=30, num_rec=10, beta=beta, sigma=0.0, seed=4,
            )
            entropies.append(InformationDynamicsSimulator(toy_data, cfg).run()["summary"]["mean_final_recommendation_entropy"])
        self.assertLess(entropies[1], entropies[0] - 0.1)

    def test_fixed_seed_reproducible(self):
        cfg = SimulationConfig(dataset="video", method="adaptive", num_users=3, num_items=500, num_iter=2, seed=12)
        first = InformationDynamicsSimulator(self.data, cfg).run()
        second = InformationDynamicsSimulator(self.data, cfg).run()
        self.assertTrue(first["per_step"].equals(second["per_step"]))
        self.assertTrue(first["final"].equals(second["final"]))

    def test_invalid_recommendation_length_rejected(self):
        with self.assertRaises(ValueError):
            SimulationConfig(dataset="video", num_items=3, num_rec=4)

    def test_bias_metrics_match_reference_definition(self):
        cfg = SimulationConfig(dataset="video", method="baseline", num_users=1, num_items=500, num_iter=1, seed=4)
        sim = InformationDynamicsSimulator(self.data, cfg)
        sim.intrinsic = np.array([[0.50, 0.30, 0.20]], dtype=float)
        rec_prob = np.array([[0.70, 0.20, 0.10]], dtype=float)
        over_mass, under_mass, over_degree, under_degree = sim._recommendation_bias_metrics(rec_prob)
        self.assertAlmostEqual(float(over_mass[0]), 0.70, places=12)
        self.assertAlmostEqual(float(under_mass[0]), 0.30, places=12)
        self.assertAlmostEqual(float(over_degree[0]), 0.40, places=12)
        self.assertAlmostEqual(float(under_degree[0]), -0.40, places=12)

    def test_news_dataset_shape(self):
        news = load_dataset("news", ROOT / "data", num_items=500, seed=4)
        self.assertEqual(news["num_topics"], 14)
        self.assertEqual(len(news["item_topics"]), 500)
        self.assertTrue(np.all(news["topic_item_counts"] > 0))

if __name__ == "__main__":
    unittest.main()
