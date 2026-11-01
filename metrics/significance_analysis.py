#!/usr/bin/env python3
"""Reproduce paired bootstrap intervals and Wilcoxon tests for ranking metrics.

Run from the repository root:
    python metrics/significance_analysis.py

The analysis reads the archived per-query results in
metrics/output/ground_truth_results.json. Misses receive rank K+1 for the
all-query mean-rank analysis. Bootstrap intervals resample the same query
indices for the hybrid and its baseline (paired percentile bootstrap).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "metrics" / "output" / "ground_truth_results.json"
OUTPUT_PATH = ROOT / "metrics" / "output" / "significance_analysis.json"
CUTOFFS = (10, 20)
BASELINES = ("tfidf", "bert")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def wilcoxon_signed_rank_normal_approximation(
    first: list[float], second: list[float]
) -> dict[str, float | int]:
    """Two-sided paired Wilcoxon signed-rank p-value with tie correction.

    Zero differences are omitted. The normal approximation is used without
    continuity correction; this matches the conventional large-sample
    approximation for the paired signed-rank statistic.
    """
    differences = [a - b for a, b in zip(first, second) if a != b]
    n = len(differences)
    if n == 0:
        return {"n_nonzero_pairs": 0, "w_positive": 0.0, "p_value_two_sided": 1.0}

    absolute = [abs(value) for value in differences]
    order = sorted(range(n), key=absolute.__getitem__)
    ranks = [0.0] * n
    tie_counts: dict[float, int] = {}
    start = 0
    while start < n:
        end = start + 1
        while end < n and absolute[order[end]] == absolute[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        tie_size = end - start
        if tie_size > 1:
            tie_counts[tie_size] = tie_counts.get(tie_size, 0) + 1
        start = end

    w_positive = sum(rank for difference, rank in zip(differences, ranks) if difference > 0)
    expected = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    variance -= sum(
        group_count * (tie_size**3 - tie_size) for tie_size, group_count in tie_counts.items()
    ) / 48.0
    if variance <= 0:
        p_value = 1.0
        z_score = 0.0
    else:
        z_score = abs(w_positive - expected) / math.sqrt(variance)
        p_value = math.erfc(z_score / math.sqrt(2.0))
    return {
        "n_nonzero_pairs": n,
        "w_positive": w_positive,
        "z_score": z_score,
        "p_value_two_sided": p_value,
    }


def paired_bootstrap_ci(
    differences: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Return the 2.5th and 97.5th percentiles of paired bootstrap means."""
    n = len(differences)
    sample_indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    bootstrap_means = differences[sample_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975], method="linear")
    return float(low), float(high)


def as_float_list(values: list[Any], description: str) -> list[float]:
    converted = [float(value) for value in values]
    if not converted:
        raise ValueError(f"No per-query values found for {description}")
    return converted


def main() -> None:
    archived = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    raw = archived["raw_results"]
    n_queries = int(archived["n_test_cases"])
    per_query: list[dict[str, Any]] = []
    metrics_by_method: dict[str, dict[int, dict[str, list[float]]]] = {
        method: {} for method in ("hybrid", "tfidf", "bert")
    }

    for method in metrics_by_method:
        for cutoff in CUTOFFS:
            result = raw[method][f"top_{cutoff}"]
            hits = [int(value) for value in result["hits"]]
            reciprocal_ranks = as_float_list(result["mrr"], f"{method} top-{cutoff} reciprocal ranks")
            retrieved_ranks = [int(value) for value in result["ranks"]]
            if not (len(hits) == len(reciprocal_ranks) == len(retrieved_ranks) == n_queries):
                raise ValueError(f"Per-query result arrays do not all contain {n_queries} queries")

            all_query_ranks: list[float] = []
            for hit, reciprocal_rank, rank in zip(hits, reciprocal_ranks, retrieved_ranks):
                if hit:
                    if rank < 1 or rank > cutoff:
                        raise ValueError(f"Invalid retrieved rank {rank} for cutoff {cutoff}")
                    if not math.isclose(reciprocal_rank, 1.0 / rank, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError("Archived reciprocal rank and rank values do not agree")
                    all_query_ranks.append(float(rank))
                else:
                    if reciprocal_rank != 0 or rank != 0:
                        raise ValueError("Archived miss should have reciprocal rank 0 and rank 0")
                    all_query_ranks.append(float(cutoff + 1))

            metrics_by_method[method][cutoff] = {
                "hit_rate": [float(value) for value in hits],
                "mrr": reciprocal_ranks,
                "all_query_mean_rank": all_query_ranks,
            }

    for query_index in range(n_queries):
        per_query.append(
            {
                "query_index_1_based": query_index + 1,
                "cutoffs": {
                    f"top_{cutoff}": {
                        method: {
                            metric: metrics_by_method[method][cutoff][metric][query_index]
                            for metric in ("hit_rate", "mrr", "all_query_mean_rank")
                        }
                        for method in metrics_by_method
                    }
                    for cutoff in CUTOFFS
                },
            }
        )

    # Keep a single RNG and a fixed loop order so the full output is reproducible.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: list[dict[str, Any]] = []
    for baseline in BASELINES:
        for cutoff in CUTOFFS:
            metric_results: dict[str, Any] = {}
            for metric in ("mrr", "hit_rate", "all_query_mean_rank"):
                hybrid_values = np.asarray(metrics_by_method["hybrid"][cutoff][metric], dtype=float)
                baseline_values = np.asarray(metrics_by_method[baseline][cutoff][metric], dtype=float)
                differences = hybrid_values - baseline_values
                ci_low, ci_high = paired_bootstrap_ci(differences, rng)
                metric_result: dict[str, Any] = {
                    "hybrid_mean": float(hybrid_values.mean()),
                    "baseline_mean": float(baseline_values.mean()),
                    "mean_difference_hybrid_minus_baseline": float(differences.mean()),
                    "paired_percentile_bootstrap_95_percent_ci": [ci_low, ci_high],
                }
                if metric == "mrr":
                    metric_result["paired_wilcoxon_signed_rank"] = (
                        wilcoxon_signed_rank_normal_approximation(
                            hybrid_values.tolist(), baseline_values.tolist()
                        )
                    )
                metric_results[metric] = metric_result
            comparisons.append(
                {
                    "cutoff": cutoff,
                    "baseline": "sentence_transformer" if baseline == "bert" else baseline,
                    "metrics": metric_results,
                }
            )

    output = {
        "analysis": "paired bootstrap CIs and Wilcoxon tests for ranking comparisons",
        "source_file": "metrics/output/ground_truth_results.json",
        "source_timestamp": archived.get("timestamp"),
        "n_queries": n_queries,
        "miss_rank_rule": "assign rank K+1 to a target not retrieved within the top K",
        "bootstrap": {
            "method": "paired percentile bootstrap resampling the same query indices for both methods",
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "seed": BOOTSTRAP_SEED,
            "percentile_method": "NumPy linear quantile",
            "rng": "numpy.random.default_rng",
            "rng_call_order": "baseline, cutoff, then MRR, hit rate, and all-query mean rank",
        },
        "wilcoxon_test": {
            "statistic": "two-sided paired Wilcoxon signed-rank test on per-query reciprocal ranks",
            "approximation": "normal approximation with tie correction and no continuity correction",
            "zero_differences": "omitted",
        },
        "per_query": per_query,
        "comparisons": comparisons,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    for comparison in comparisons:
        cutoff = comparison["cutoff"]
        baseline = comparison["baseline"]
        mrr = comparison["metrics"]["mrr"]
        hr = comparison["metrics"]["hit_rate"]
        rank = comparison["metrics"]["all_query_mean_rank"]
        p_value = mrr["paired_wilcoxon_signed_rank"]["p_value_two_sided"]
        print(
            f"Top-{cutoff} vs {baseline}: "
            f"MRR delta={mrr['mean_difference_hybrid_minus_baseline']:+.3f} "
            f"CI=[{mrr['paired_percentile_bootstrap_95_percent_ci'][0]:.3f}, "
            f"{mrr['paired_percentile_bootstrap_95_percent_ci'][1]:.3f}], p={p_value:.6g}; "
            f"HR delta={hr['mean_difference_hybrid_minus_baseline']:+.2f} "
            f"CI=[{hr['paired_percentile_bootstrap_95_percent_ci'][0]:.2f}, "
            f"{hr['paired_percentile_bootstrap_95_percent_ci'][1]:.2f}]; "
            f"all-query mean-rank delta={rank['mean_difference_hybrid_minus_baseline']:+.2f} "
            f"CI=[{rank['paired_percentile_bootstrap_95_percent_ci'][0]:.2f}, "
            f"{rank['paired_percentile_bootstrap_95_percent_ci'][1]:.2f}]"
        )


if __name__ == "__main__":
    main()
