"""Evaluation modules for perplexity, memory, and latency profiling."""

from evaluation.perplexity import compute_perplexity
from evaluation.memory import MemoryProfiler, profile_memory, calculate_kv_cache_bytes
from evaluation.latency import LatencyBenchmarker, benchmark_latency
from evaluation.forecast_metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_block_selection_accuracy,
    compute_kl_divergence,
    evaluate_forecast_metrics,
)

__all__ = [
    "compute_perplexity",
    "MemoryProfiler",
    "profile_memory",
    "calculate_kv_cache_bytes",
    "LatencyBenchmarker",
    "benchmark_latency",
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_block_selection_accuracy",
    "compute_kl_divergence",
    "evaluate_forecast_metrics",
]

