from __future__ import annotations

import csv
import math
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


@dataclass
class RequestMetric:
    success: bool
    latency_s: float
    ttft_s: float
    tpot_s: float
    target_input_tokens: int
    user_content_tokens: int
    final_prompt_tokens: int
    prompt_tokens: int
    max_tokens: int
    server_max_tokens: int
    output_tokens: int
    finished_at_s: float


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._items: List[RequestMetric] = []
        self._inflight = 0
        self._peak_inflight = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        # Approximation for model-agnostic benchmarking.
        return len(text.split())

    def add(self, metric: RequestMetric) -> None:
        with self._lock:
            self._items.append(metric)

    def register_request_start(self) -> None:
        with self._lock:
            self._inflight += 1
            if self._inflight > self._peak_inflight:
                self._peak_inflight = self._inflight

    def register_request_end(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        if p <= 0:
            return min(values)
        if p >= 100:
            return max(values)
        ordered = sorted(values)
        idx = (len(ordered) - 1) * (p / 100.0)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        weight = idx - lo
        return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

    def summary(self, concurrency: int, request_rate: float) -> dict:
        with self._lock:
            items = list(self._items)
            peak_inflight = self._peak_inflight
        elapsed = max(time.time() - self._started_at, 1e-9)
        ok = [m for m in items if m.success]
        success_count = len(ok)
        total = len(items)
        fail_count = total - success_count
        input_tokens = sum(m.prompt_tokens for m in ok)
        avg_target_input_tokens = _safe_mean([float(m.target_input_tokens) for m in ok])
        avg_user_content_tokens = _safe_mean([float(m.user_content_tokens) for m in ok])
        avg_final_prompt_tokens = _safe_mean([float(m.final_prompt_tokens) for m in ok])
        avg_prompt_tokens = _safe_mean([float(m.prompt_tokens) for m in ok])
        avg_max_tokens = _safe_mean([float(m.max_tokens) for m in ok])
        avg_server_max_tokens = _safe_mean([float(m.server_max_tokens) for m in ok])
        output_tokens = sum(m.output_tokens for m in ok)
        total_tokens = input_tokens + output_tokens
        ttft_values = [m.ttft_s for m in ok]
        latency_values = [m.latency_s for m in ok]
        tpot_values = [m.tpot_s for m in ok]

        peak_output_tokens_per_s = 0.0
        if ok:
            buckets = {}
            for m in ok:
                rel_sec = int(max(0.0, m.finished_at_s - self._started_at))
                buckets[rel_sec] = buckets.get(rel_sec, 0) + m.output_tokens
            peak_output_tokens_per_s = max(buckets.values()) if buckets else 0.0

        avg_concurrency = (sum(latency_values) / elapsed) if latency_values else 0.0

        return {
            "concurrency": concurrency,
            "request_rate": request_rate,
            "input_tokens": input_tokens,
            "target_input_tokens": round(avg_target_input_tokens, 2),
            "user_content_tokens": round(avg_user_content_tokens, 2),
            "final_prompt_tokens": round(avg_final_prompt_tokens, 2),
            "actual_prompt_tokens": round(avg_final_prompt_tokens, 2),
            "prompt_tokens": round(avg_prompt_tokens, 2),
            "max_tokens": round(avg_max_tokens, 2),
            "server_max_tokens": round(avg_server_max_tokens, 2),
            "output_tokens": output_tokens,
            "success_count": success_count,
            "fail_count": fail_count,
            "avg_ttft": _safe_mean([m.ttft_s for m in ok]),
            "avg_tpot": _safe_mean([m.tpot_s for m in ok]),
            "avg_latency": _safe_mean([m.latency_s for m in ok]),
            "output_tokens_per_s": output_tokens / elapsed,
            "total_tokens_per_s": total_tokens / elapsed,
            "request_throughput": success_count / elapsed,
            "peak_output_tokens_per_s": peak_output_tokens_per_s,
            "peak_concurrent_requests": peak_inflight,
            "avg_concurrency": avg_concurrency,
            "ttft_p50": self._percentile(ttft_values, 50),
            "ttft_p99": self._percentile(ttft_values, 99),
            "latency_p50": self._percentile(latency_values, 50),
            "latency_p99": self._percentile(latency_values, 99),
            "tpot_p50": self._percentile(tpot_values, 50),
            "tpot_p99": self._percentile(tpot_values, 99),
            "elapsed_s": elapsed,
        }

    def write_summary_csv(
        self,
        output_path: str,
        summary_data: dict,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output.exists()
        with open(output, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_data.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary_data)
        return output


def format_benchmark_report(summary: dict) -> str:
    lines = [
        "============ Serving Benchmark Result ============",
        f"Traffic request rate:                    {summary.get('request_rate', 0)}",
        f"Max request concurrency:                 {summary.get('concurrency', 0)}",
        f"Successful requests:                     {summary.get('success_count', 0)}",
        f"Failed requests:                         {summary.get('fail_count', 0)}",
        f"Benchmark duration (s):                  {summary.get('elapsed_s', 0):.2f}",
        f"Total input tokens:                      {summary.get('input_tokens', 0)}",
        f"Average prompt tokens:                   {summary.get('prompt_tokens', 0)}",
        f"Average max tokens:                      {summary.get('max_tokens', 0)}",
        f"Total generated tokens:                  {summary.get('output_tokens', 0)}",
        f"Request throughput (req/s):              {summary.get('request_throughput', 0):.2f}",
        f"Input token throughput (tok/s):          {(summary.get('total_tokens_per_s', 0) - summary.get('output_tokens_per_s', 0)):.2f}",
        f"Output token throughput (tok/s):         {summary.get('output_tokens_per_s', 0):.2f}",
        f"Peak output token throughput (tok/s):    {summary.get('peak_output_tokens_per_s', 0):.2f}",
        f"Peak concurrent requests:                {summary.get('peak_concurrent_requests', 0)}",
        f"Total token throughput (tok/s):          {summary.get('total_tokens_per_s', 0):.2f}",
        f"Concurrency:                             {summary.get('avg_concurrency', 0):.2f}",
        "----------------End-to-End Latency----------------",
        f"Mean E2E Latency (ms):                   {summary.get('avg_latency', 0) * 1000:.2f}",
        f"Median E2E Latency (ms):                 {summary.get('latency_p50', 0) * 1000:.2f}",
        f"P99 E2E Latency (ms):                    {summary.get('latency_p99', 0) * 1000:.2f}",
        "---------------Time to First Token----------------",
        f"Mean TTFT (ms):                          {summary.get('avg_ttft', 0) * 1000:.2f}",
        f"Median TTFT (ms):                        {summary.get('ttft_p50', 0) * 1000:.2f}",
        f"P99 TTFT (ms):                           {summary.get('ttft_p99', 0) * 1000:.2f}",
        "-----Time per Output Token (excl. 1st token)------",
        f"Mean TPOT (ms):                          {summary.get('avg_tpot', 0) * 1000:.2f}",
        f"Median TPOT (ms):                        {summary.get('tpot_p50', 0) * 1000:.2f}",
        f"P99 TPOT (ms):                           {summary.get('tpot_p99', 0) * 1000:.2f}",
        "==================================================",
    ]
    return "\n".join(lines)
