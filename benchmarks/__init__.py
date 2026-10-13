"""Benchmarks suite for SparDA research model."""

from benchmarks.dense import run_dense_benchmarks
from benchmarks.sparse import run_comparative_benchmark

__all__ = ["run_dense_benchmarks", "run_comparative_benchmark"]

