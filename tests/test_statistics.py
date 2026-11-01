"""Recompute the existing published summaries from their seed-level rows."""

import csv
from pathlib import Path
import unittest

from spcp.statistics import average_hm, hm, summarize_seed_scores, target_average

ROOT = Path(__file__).resolve().parents[1]


class StatisticsChecks(unittest.TestCase):
    def test_published_seed_aggregation(self):
        with (ROOT / "results/full_method_per_seed.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        with (ROOT / "results/full_method_mean_sd.csv").open(encoding="utf-8") as stream:
            summaries = list(csv.DictReader(stream))
        groups = {}
        for row in rows:
            key = (row["task"], row["domain"], row["metric"])
            groups.setdefault(key, {})[int(row["seed"])] = float(row["accuracy"])
        for summary in summaries:
            key = (summary["task"], summary["domain"], summary["metric"])
            if key == ("b2n", "AverageHM", "hm"):
                inputs = [
                    {
                        "dataset": domain,
                        "seed": seed,
                        "base": value,
                        "new": groups[("b2n", domain, "new")][seed],
                    }
                    for (task, domain, metric), values in groups.items()
                    if task == "b2n" and metric == "base"
                    for seed, value in values.items()
                ]
                result = average_hm(inputs)
            elif summary["domain"] == "Avg":
                task = summary["task"]
                domains = [
                    domain
                    for t, domain, metric in groups
                    if t == task and metric == "accuracy" and domain != "Avg"
                ]
                inputs = [
                    {"domain": domain, "seed": seed, "accuracy": value}
                    for domain in domains
                    for seed, value in groups[(task, domain, "accuracy")].items()
                ]
                result = target_average(inputs, domains)
            else:
                result = summarize_seed_scores(groups[key])
            with self.subTest(summary=key):
                self.assertAlmostEqual(result["mean"], float(summary["mean"]), places=10)
                self.assertAlmostEqual(result["sample_sd"], float(summary["sample_sd"]), places=10)
        for (task, domain, metric), values in groups.items():
            if task == "b2n" and metric == "hm" and domain != "AverageHM":
                for seed, value in values.items():
                    self.assertAlmostEqual(
                        value,
                        hm(
                            groups[(task, domain, "base")][seed],
                            groups[(task, domain, "new")][seed],
                        ),
                        places=10,
                    )


if __name__ == "__main__":
    unittest.main()
