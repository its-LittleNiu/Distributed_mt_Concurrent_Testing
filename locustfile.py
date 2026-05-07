from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import gevent
from gevent.lock import Semaphore
from locust import HttpUser, events, task
from locust.runners import LocalRunner, MasterRunner, WorkerRunner

from bench.dataset import PromptProvider
from bench.metrics import MetricsCollector, RequestMetric, format_benchmark_report, summary_from_snapshots


BENCH_FINAL_SNAPSHOT_MESSAGE = "bench_final_snapshot"

metrics = MetricsCollector()
prompt_provider: Optional[PromptProvider] = None
resolved_model: str = ""
resolved_tokenizer = None
cpu_bottleneck_detected = False
_cpu_monitor_stop = threading.Event()
_cpu_monitor_thread: Optional[threading.Thread] = None
_worker_final_snapshot_sent = False


@dataclass
class DistributedRunState:
    connected_workers: set[str] = field(default_factory=set)
    latest_reports: dict[str, dict] = field(default_factory=dict)
    final_snapshots: dict[str, dict] = field(default_factory=dict)
    worker_cpu_flags: dict[str, bool] = field(default_factory=dict)
    expected_workers: int = 0
    finalized: bool = False
    last_result: Optional[dict] = None

    def reset(self, expected_workers: int, connected_workers: set[str]) -> None:
        self.connected_workers = set(connected_workers)
        self.latest_reports = {}
        self.final_snapshots = {}
        self.worker_cpu_flags = {}
        self.expected_workers = expected_workers
        self.finalized = False
        self.last_result = None


distributed_run_state = DistributedRunState()
last_run_result: Optional[dict] = None


class RequestRateLimiter:
    def __init__(self) -> None:
        self._lock = Semaphore()
        self._rate_per_s = 0.0
        self._next_start_at = 0.0

    def reset(self, rate_per_s: float) -> None:
        with self._lock:
            self._rate_per_s = max(0.0, float(rate_per_s or 0.0))
            self._next_start_at = time.perf_counter()

    def wait(self) -> None:
        with self._lock:
            rate_per_s = self._rate_per_s
            if rate_per_s <= 0:
                return
            interval_s = 1.0 / rate_per_s
            now = time.perf_counter()
            scheduled_start = max(now, self._next_start_at)
            self._next_start_at = scheduled_start + interval_s
            sleep_s = scheduled_start - now

        if sleep_s > 0:
            gevent.sleep(sleep_s)


request_rate_limiter = RequestRateLimiter()


def _is_master(environment) -> bool:
    return isinstance(environment.runner, MasterRunner)


def _is_worker(environment) -> bool:
    return isinstance(environment.runner, WorkerRunner)


def _parse_run_time_seconds(raw: str) -> float:
    text = (raw or "").strip().lower()
    if not text:
        return 0.0
    if text.replace(".", "", 1).isdigit():
        return float(text)

    total = 0.0
    position = 0
    for match in re.finditer(r"(\d+(?:\.\d+)?)([hms])", text):
        if match.start() != position:
            return 0.0
        value = float(match.group(1))
        unit = match.group(2)
        total += value * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
        position = match.end()
    if position != len(text):
        return 0.0
    return total


def _labeled_concurrency(environment) -> int:
    opts = environment.parsed_options
    labeled = int(getattr(opts, "benchmark_concurrency", 0) or 0)
    if labeled <= 0:
        labeled = int(getattr(opts, "num_users", 0) or 0)
    if labeled <= 0 and environment.runner:
        labeled = int(getattr(environment.runner, "target_user_count", 0) or 0)
    if labeled <= 0 and environment.runner:
        labeled = int(getattr(environment.runner, "user_count", 0) or 0)
    return labeled


def _summary_base(environment, summary: dict, client_cpu_bottleneck: bool) -> dict:
    opts = environment.parsed_options
    summary = dict(summary)
    summary.update(
        {
            "host": environment.host,
            "model": resolved_model or opts.model,
            "client_cpu_bottleneck": str(client_cpu_bottleneck).lower(),
        }
    )
    return summary


def _validation_reasons(
    environment,
    summary: dict,
    connected_workers: Optional[set[str]] = None,
    final_snapshot_workers: Optional[set[str]] = None,
    missing_workers: Optional[set[str]] = None,
) -> list[str]:
    opts = environment.parsed_options
    reasons: list[str] = []
    total_requests = int(summary.get("success_count", 0)) + int(summary.get("fail_count", 0))
    if total_requests <= 0:
        reasons.append("no requests were completed in this run")

    expected_run_time_s = _parse_run_time_seconds(getattr(opts, "run_time", ""))
    elapsed_s = float(summary.get("elapsed_s", 0.0) or 0.0)
    if expected_run_time_s > 0:
        min_elapsed_s = max(expected_run_time_s - 2.0, expected_run_time_s * 0.95)
        if elapsed_s < min_elapsed_s:
            reasons.append(
                f"run stopped early: elapsed_s={elapsed_s:.2f}, expected_run_time_s={expected_run_time_s:.2f}"
            )

    expected_workers = int(getattr(opts, "expect_workers", 0) or 0)
    if connected_workers is not None and expected_workers > 0 and len(connected_workers) < expected_workers:
        reasons.append(
            f"only {len(connected_workers)} workers connected but expect_workers={expected_workers}"
        )
    if missing_workers:
        reasons.append("workers went missing during the run: " + ", ".join(sorted(missing_workers)))
    if connected_workers is not None and final_snapshot_workers is not None:
        missing_final = (set(connected_workers) - set(missing_workers or set())) - set(final_snapshot_workers)
        if missing_final:
            reasons.append(
                "master did not receive final metrics from workers: " + ", ".join(sorted(missing_final))
            )
    return reasons


def _format_invalid_report(summary: dict, reasons: list[str]) -> str:
    lines = [
        "============ Serving Benchmark Result ============",
        "Result status:                           invalid",
        f"Successful requests:                     {summary.get('success_count', 0)}",
        f"Failed requests:                         {summary.get('fail_count', 0)}",
        f"Benchmark duration (s):                  {summary.get('elapsed_s', 0):.2f}",
        "Invalid reasons:",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("==================================================")
    return "\n".join(lines)


def _write_result_json(path: str, payload: dict) -> None:
    if not path:
        return
    output = os.path.abspath(path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _persist_run_result(environment, summary: dict, valid: bool, reasons: list[str]) -> dict:
    opts = environment.parsed_options
    result = {
        "status": "ok" if valid else "invalid",
        "valid": valid,
        "reasons": reasons,
        "summary": summary,
        "distributed": _is_master(environment) or _is_worker(environment),
        "timestamp": time.time(),
    }
    _write_result_json(getattr(opts, "result_json", ""), result)
    if valid:
        result_file = metrics.write_summary_csv(opts.summary_csv, summary)
        print(f"[bench] summary written: {result_file}")
        print(format_benchmark_report(summary))
    else:
        print("[bench] summary skipped because run is invalid")
        print(_format_invalid_report(summary, reasons))
    if summary.get("client_cpu_bottleneck") == "true":
        print("[bench] warning: client CPU >= 90% detected during this run")
    return result


def _worker_ids_from_runner(environment) -> set[str]:
    runner = environment.runner
    if not isinstance(runner, MasterRunner):
        return set()
    return {worker.id for worker in runner.clients.all}


def _missing_worker_ids(environment) -> set[str]:
    runner = environment.runner
    if not isinstance(runner, MasterRunner):
        return set()
    return {worker.id for worker in runner.clients.missing}


def _send_final_worker_snapshot(environment) -> None:
    global _worker_final_snapshot_sent
    if _worker_final_snapshot_sent or not _is_worker(environment):
        return
    environment.runner.send_message(
        BENCH_FINAL_SNAPSHOT_MESSAGE,
        {
            "snapshot": metrics.snapshot(),
            "cpu_bottleneck_detected": cpu_bottleneck_detected,
        },
    )
    _worker_final_snapshot_sent = True


def _wait_for_final_worker_snapshots(environment, timeout_s: float = 5.0) -> None:
    if not _is_master(environment):
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        expected_final_workers = _worker_ids_from_runner(environment) - _missing_worker_ids(environment)
        if expected_final_workers.issubset(distributed_run_state.final_snapshots.keys()):
            return
        gevent.sleep(0.2)


def _finalize_local_run(environment) -> dict:
    summary = metrics.summary(
        concurrency=_labeled_concurrency(environment),
        request_rate=environment.parsed_options.request_rate,
    )
    summary = _summary_base(environment, summary, cpu_bottleneck_detected)
    reasons = _validation_reasons(environment, summary)
    return _persist_run_result(environment, summary, not reasons, reasons)


def _finalize_master_run(environment) -> dict:
    if distributed_run_state.finalized and distributed_run_state.last_result is not None:
        return distributed_run_state.last_result

    connected_workers = _worker_ids_from_runner(environment) or set(distributed_run_state.connected_workers)
    final_snapshot_workers = set(distributed_run_state.final_snapshots.keys())
    missing_workers = _missing_worker_ids(environment)
    snapshots = [distributed_run_state.final_snapshots[worker_id] for worker_id in sorted(final_snapshot_workers)]
    summary = summary_from_snapshots(
        snapshots=snapshots,
        concurrency=_labeled_concurrency(environment),
        request_rate=environment.parsed_options.request_rate,
    )
    summary = _summary_base(environment, summary, any(distributed_run_state.worker_cpu_flags.values()))
    reasons = _validation_reasons(
        environment,
        summary,
        connected_workers=connected_workers,
        final_snapshot_workers=final_snapshot_workers,
        missing_workers=missing_workers,
    )
    result = _persist_run_result(environment, summary, not reasons, reasons)
    distributed_run_state.finalized = True
    distributed_run_state.last_result = result
    return result


def _start_cpu_monitor() -> None:
    global cpu_bottleneck_detected, _cpu_monitor_thread
    try:
        import psutil
    except Exception:
        print("[bench] warning: psutil unavailable, cpu bottleneck flag disabled")
        return
    cpu_bottleneck_detected = False
    _cpu_monitor_stop.clear()

    def _run():
        global cpu_bottleneck_detected
        while not _cpu_monitor_stop.is_set():
            if psutil.cpu_percent(interval=1.0) >= 90.0:
                cpu_bottleneck_detected = True

    _cpu_monitor_thread = threading.Thread(target=_run, daemon=True)
    _cpu_monitor_thread.start()


def _stop_cpu_monitor() -> None:
    _cpu_monitor_stop.set()


def _resolve_model_id(host: str) -> str:
    url = f"{host.rstrip('/')}/v1/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, OSError):
        return ""
    try:
        data = json.loads(raw)
        models = data.get("data", [])
        if isinstance(models, list) and models and isinstance(models[0], dict):
            return str(models[0].get("id", "") or "")
    except Exception:
        return ""
    return ""


@events.init.add_listener
def _(environment, **kwargs):
    if isinstance(environment.runner, MasterRunner):
        environment.runner.register_message(
            BENCH_FINAL_SNAPSHOT_MESSAGE,
            _on_worker_final_snapshot,
            concurrent=True,
        )


def _on_worker_final_snapshot(environment, msg, **kwargs):
    client_id = getattr(msg, "node_id", None) or getattr(msg, "client_id", None) or "unknown"
    payload = msg.data or {}
    distributed_run_state.connected_workers.add(client_id)
    distributed_run_state.final_snapshots[client_id] = payload.get("snapshot", {})
    distributed_run_state.worker_cpu_flags[client_id] = bool(payload.get("cpu_bottleneck_detected", False))


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--dataset", type=str, default="", help="ShareGPT json path")
    parser.add_argument(
        "--benchmark-concurrency",
        type=int,
        default=0,
        help="Explicit concurrency label used in summary csv",
    )
    parser.add_argument(
        "--input-output",
        type=str,
        default="[4000:1000],[6000:1000]",
        help="Length pairs like [4000:1000],[6000:1000]",
    )
    parser.add_argument("--model", type=str, default="", help="Optional model id")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="HF tokenizer path or local tokenizer directory",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Fallback max_tokens")
    parser.add_argument(
        "--server-max-tokens",
        type=int,
        default=8192,
        help="Backend max context tokens limit",
    )
    parser.add_argument(
        "--prompt-token-reserve",
        type=int,
        default=128,
        help="Safety reserve tokens for system/payload overhead",
    )
    parser.add_argument(
        "--prompt-budget-ratio",
        type=float,
        default=0.6,
        help="Conservative ratio applied to prompt budget due to tokenizer mismatch",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Target HTTP request start rate per Locust process. Use <=0 for unlimited.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/summary.csv",
        help="Summary csv file path",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default="",
        help="Optional per-run json output for matrix orchestration",
    )


@events.worker_connect.add_listener
def _(client_id, **kwargs):
    distributed_run_state.connected_workers.add(client_id)


@events.worker_report.add_listener
def _(client_id, data, **kwargs):
    distributed_run_state.connected_workers.add(client_id)
    meta = data.get("bench_meta", {})
    distributed_run_state.latest_reports[client_id] = meta
    if "cpu_bottleneck_detected" in meta:
        distributed_run_state.worker_cpu_flags[client_id] = bool(meta.get("cpu_bottleneck_detected", False))


@events.report_to_master.add_listener
def _(client_id, data, **kwargs):
    data["bench_meta"] = {
        **metrics.status(),
        "cpu_bottleneck_detected": cpu_bottleneck_detected,
    }


@events.test_start.add_listener
def _(environment, **kwargs):
    global prompt_provider, resolved_model, resolved_tokenizer, _worker_final_snapshot_sent, last_run_result

    metrics.reset()
    _worker_final_snapshot_sent = False
    last_run_result = None
    _stop_cpu_monitor()

    if _is_master(environment):
        distributed_run_state.reset(
            expected_workers=int(getattr(environment.parsed_options, "expect_workers", 0) or 0),
            connected_workers=_worker_ids_from_runner(environment),
        )
        return

    opts = environment.parsed_options
    prompt_provider = PromptProvider(
        dataset_path=opts.dataset or None,
        input_output_pairs=opts.input_output,
        seed=42,
    )
    resolved_model = opts.model or _resolve_model_id(environment.host or "")
    if not resolved_model:
        raise RuntimeError(
            "model is required. Please pass --model <model_id>, "
            "or ensure GET /v1/models is available for auto-discovery."
        )

    tokenizer_source = opts.tokenizer_path or resolved_model
    try:
        from transformers import AutoTokenizer

        resolved_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"[bench] tokenizer loaded from: {tokenizer_source}")
    except Exception as exc:
        try:
            from transformers import PreTrainedTokenizerFast

            tokenizer_file = os.path.join(tokenizer_source, "tokenizer.json")
            resolved_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            print(f"[bench] tokenizer loaded from tokenizer.json: {tokenizer_file}")
        except Exception as exc2:
            raise RuntimeError(
                "Tokenizer load failed in strict shaping mode. "
                f"source={tokenizer_source}, auto_err={exc}, fast_err={exc2}"
            )

    request_rate_limiter.reset(float(opts.request_rate))
    if float(opts.request_rate) > 0:
        print(f"[bench] request rate limiter enabled: {float(opts.request_rate):.4f} req/s")
    else:
        print("[bench] request rate limiter disabled: unlimited request start rate")
    _start_cpu_monitor()


@events.test_stop.add_listener
def _(environment, **kwargs):
    global last_run_result
    _stop_cpu_monitor()
    if _is_worker(environment):
        _send_final_worker_snapshot(environment)
        return
    if _is_master(environment):
        return
    last_run_result = _finalize_local_run(environment)


@events.quitting.add_listener
def _(environment, **kwargs):
    global last_run_result
    if _is_worker(environment):
        _send_final_worker_snapshot(environment)
        return
    if _is_master(environment):
        _wait_for_final_worker_snapshots(environment)
        last_run_result = _finalize_master_run(environment)
    if last_run_result is not None:
        environment.process_exit_code = 0 if last_run_result.get("valid") else 2


class TRTLLMUser(HttpUser):
    wait_time = lambda self: 0  # noqa: E731

    @staticmethod
    def _ensure_tokenizer_available() -> None:
        if resolved_tokenizer is None:
            raise RuntimeError(
                "Tokenizer is required for strict input/output shaping. "
                "Please provide a valid --tokenizer-path."
            )

    @staticmethod
    def _build_prompt_with_exact_tokens(base_prompt: str, target_input_tokens: int) -> tuple[str, int, int, int]:
        TRTLLMUser._ensure_tokenizer_available()
        if target_input_tokens <= 0:
            return "", 0, 0, 0

        base = (base_prompt or "").strip() or "hello"
        base_ids = resolved_tokenizer.encode(base, add_special_tokens=False)
        if len(base_ids) >= target_input_tokens:
            expanded = base_ids[:target_input_tokens]
        else:
            piece = "\n" + base
            piece_ids = resolved_tokenizer.encode(piece, add_special_tokens=False)
            if not piece_ids:
                piece = " " + base
                piece_ids = resolved_tokenizer.encode(piece, add_special_tokens=False)
            if not piece_ids:
                raise RuntimeError("Tokenizer produced empty piece ids for prompt expansion")

            expanded = list(base_ids)
            remaining = target_input_tokens - len(expanded)
            whole_repeats, tail = divmod(remaining, len(piece_ids))
            if whole_repeats > 0:
                expanded.extend(piece_ids * whole_repeats)
            if tail > 0:
                expanded.extend(piece_ids[:tail])

        decoded = resolved_tokenizer.decode(expanded, skip_special_tokens=False)
        reencoded = resolved_tokenizer.encode(decoded, add_special_tokens=False)
        if len(reencoded) < target_input_tokens:
            padding_ids = resolved_tokenizer.encode("\n" + base, add_special_tokens=False) or expanded[-32:]
            for _ in range(4):
                if len(reencoded) >= target_input_tokens:
                    break
                deficit = target_input_tokens - len(reencoded)
                expanded.extend(padding_ids[:deficit])
                gevent.sleep(0)
                decoded = resolved_tokenizer.decode(expanded, skip_special_tokens=False)
                reencoded = resolved_tokenizer.encode(decoded, add_special_tokens=False)

        return decoded, len(reencoded), len(expanded), len(expanded)

    @staticmethod
    def _truncate_prompt_to_budget(prompt: str, token_budget: int) -> tuple[str, int]:
        TRTLLMUser._ensure_tokenizer_available()
        if token_budget <= 0:
            return "", 0
        token_ids = resolved_tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= token_budget:
            return prompt, len(token_ids)
        token_ids = token_ids[:token_budget]
        return resolved_tokenizer.decode(token_ids, skip_special_tokens=True), len(token_ids)

    @staticmethod
    def _count_chat_prompt_tokens(messages: list[dict]) -> int:
        TRTLLMUser._ensure_tokenizer_available()
        if hasattr(resolved_tokenizer, "apply_chat_template"):
            try:
                ids = resolved_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return len(ids)
            except Exception:
                pass
        joined = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
        ids = resolved_tokenizer.encode(joined, add_special_tokens=True)
        return len(ids) + 256

    @task
    def chat_completion_stream(self):
        if prompt_provider is None:
            return

        opts = self.environment.parsed_options
        request_name = "chat_completions_stream"
        max_build_retries = 5
        safe_prompt = ""
        user_content_tokens = 0
        expanded_len = 0
        encoded_len = 0
        target_input_tokens = 0
        max_tokens = 0

        last_build_error = ""
        for attempt in range(1, max_build_retries + 1):
            sample = prompt_provider.sample()
            max_tokens = int(sample.target_output_tokens or opts.max_tokens)
            target_input_tokens = int(sample.target_input_tokens)
            if opts.prompt_budget_ratio > 0:
                target_input_tokens = int(target_input_tokens * float(opts.prompt_budget_ratio))
                target_input_tokens = max(target_input_tokens, 1)
            sample_hash = hashlib.sha1(sample.prompt.encode("utf-8", errors="ignore")).hexdigest()[:12]
            safe_prompt, user_content_tokens, expanded_len, encoded_len = self._build_prompt_with_exact_tokens(
                sample.prompt,
                target_input_tokens,
            )
            print(
                "[dbg] prompt_shape "
                f"attempt={attempt}, sample_hash={sample_hash}, "
                f"target_input_tokens={target_input_tokens}, "
                f"len_base_prompt={len(sample.prompt)}, "
                f"len_token_ids={encoded_len}, "
                f"len_expanded={expanded_len}, "
                f"len_decoded_text={len(safe_prompt)}, "
                f"len_reencoded={user_content_tokens}"
            )
            min_required = int(target_input_tokens * 0.9)
            if user_content_tokens >= min_required:
                break
            last_build_error = (
                "Tokenizer decode/encode collapsed prompt unexpectedly. "
                f"attempt={attempt}, target_input_tokens={target_input_tokens}, "
                f"expanded={expanded_len}, reencoded={user_content_tokens}, "
                f"required_min={min_required}, len_base_prompt={len(sample.prompt)}, "
                f"sample_hash={sample_hash}"
            )
            print(f"[dbg] shape_retry {last_build_error}")
            gevent.sleep(0)
        else:
            raise RuntimeError(f"{last_build_error} (retries_exhausted={max_build_retries})")

        server_max_tokens = int(opts.server_max_tokens)
        allowed_prompt_tokens = server_max_tokens - int(opts.prompt_token_reserve) - max_tokens
        prompt_tokens = user_content_tokens
        if allowed_prompt_tokens <= 0:
            metrics.add(
                RequestMetric(
                    success=False,
                    latency_s=0.0,
                    ttft_s=0.0,
                    tpot_s=0.0,
                    target_input_tokens=target_input_tokens,
                    user_content_tokens=0,
                    final_prompt_tokens=0,
                    prompt_tokens=0,
                    max_tokens=max_tokens,
                    server_max_tokens=server_max_tokens,
                    output_tokens=0,
                    finished_at_s=time.time(),
                )
            )
            print(
                "[dbg] invalid budget "
                f"max_tokens={max_tokens}, reserve={opts.prompt_token_reserve}, "
                f"server_max_tokens={server_max_tokens}"
            )
            return

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": safe_prompt},
        ]
        final_prompt_tokens = self._count_chat_prompt_tokens(messages)
        while final_prompt_tokens > allowed_prompt_tokens and user_content_tokens > 0:
            overflow = final_prompt_tokens - allowed_prompt_tokens
            trim_step = max(overflow + 16, int(user_content_tokens * 0.05), 32)
            next_user_budget = max(0, user_content_tokens - trim_step)
            safe_prompt, user_content_tokens = self._truncate_prompt_to_budget(safe_prompt, next_user_budget)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": safe_prompt},
            ]
            final_prompt_tokens = self._count_chat_prompt_tokens(messages)
            gevent.sleep(0)

        prompt_tokens = final_prompt_tokens
        print(
            "[dbg] prompt_final "
            f"target_input_tokens={target_input_tokens}, "
            f"user_content_tokens={user_content_tokens}, "
            f"final_prompt_tokens={final_prompt_tokens}, "
            f"prompt_tokens_finally_recorded={prompt_tokens}"
        )
        min_prompt_threshold = int(target_input_tokens * 0.9)
        if final_prompt_tokens < min_prompt_threshold:
            raise RuntimeError(
                "Prompt too short for strict shaping. "
                f"target_input_tokens={target_input_tokens}, "
                f"final_prompt_tokens={final_prompt_tokens}, "
                f"required_min={min_prompt_threshold}"
            )
        if final_prompt_tokens > allowed_prompt_tokens:
            metrics.add(
                RequestMetric(
                    success=False,
                    latency_s=0.0,
                    ttft_s=0.0,
                    tpot_s=0.0,
                    target_input_tokens=target_input_tokens,
                    user_content_tokens=user_content_tokens,
                    final_prompt_tokens=final_prompt_tokens,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    server_max_tokens=server_max_tokens,
                    output_tokens=0,
                    finished_at_s=time.time(),
                )
            )
            print(
                "[dbg] local_reject "
                f"target_input_tokens={target_input_tokens}, "
                f"user_content_tokens={user_content_tokens}, "
                f"final_prompt_tokens={final_prompt_tokens}, "
                f"max_tokens={max_tokens}, server_max_tokens={server_max_tokens}"
            )
            return

        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        if resolved_model:
            payload["model"] = resolved_model

        request_rate_limiter.wait()
        start = time.perf_counter()
        first_token_at = None
        output_fragments = []
        chunks_count = 0
        metrics.register_request_start()

        try:
            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=3600,
                catch_response=True,
                name=request_name,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}: {response.text[:256]}")
                    metrics.add(
                        RequestMetric(
                            success=False,
                            latency_s=time.perf_counter() - start,
                            ttft_s=0.0,
                            tpot_s=0.0,
                            target_input_tokens=target_input_tokens,
                            user_content_tokens=user_content_tokens,
                            final_prompt_tokens=final_prompt_tokens,
                            prompt_tokens=prompt_tokens,
                            max_tokens=max_tokens,
                            server_max_tokens=server_max_tokens,
                            output_tokens=0,
                            finished_at_s=time.time(),
                        )
                    )
                    print(
                        "[dbg] fail "
                        f"status={response.status_code}, target_input_tokens={target_input_tokens}, "
                        f"user_content_tokens={user_content_tokens}, final_prompt_tokens={final_prompt_tokens}, "
                        f"max_tokens={max_tokens}, server_max_tokens={server_max_tokens}, "
                        f"body={response.text[:200]}"
                    )
                    return

                try:
                    for raw in response.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            output_fragments.append(delta)
                            chunks_count += 1
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                except Exception as exc:
                    response.failure(f"stream parse error: {exc}")
                    metrics.add(
                        RequestMetric(
                            success=False,
                            latency_s=time.perf_counter() - start,
                            ttft_s=0.0,
                            tpot_s=0.0,
                            target_input_tokens=target_input_tokens,
                            user_content_tokens=user_content_tokens,
                            final_prompt_tokens=final_prompt_tokens,
                            prompt_tokens=prompt_tokens,
                            max_tokens=max_tokens,
                            server_max_tokens=server_max_tokens,
                            output_tokens=0,
                            finished_at_s=time.time(),
                        )
                    )
                    return

                response.success()
        finally:
            metrics.register_request_end()

        end = time.perf_counter()
        ttft = (first_token_at - start) if first_token_at else 0.0
        latency = end - start
        output_text = "".join(output_fragments)
        output_tokens = metrics.estimate_tokens(output_text)
        if output_tokens <= 0 and chunks_count > 0:
            output_tokens = chunks_count
        tpot = (latency - ttft) / max(output_tokens - 1, 1) if output_tokens > 0 else 0.0

        metrics.add(
            RequestMetric(
                success=True,
                latency_s=latency,
                ttft_s=ttft,
                tpot_s=tpot,
                target_input_tokens=target_input_tokens,
                user_content_tokens=user_content_tokens,
                final_prompt_tokens=final_prompt_tokens,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                server_max_tokens=server_max_tokens,
                output_tokens=output_tokens,
                finished_at_s=time.time(),
            )
        )
