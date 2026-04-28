from __future__ import annotations

import json
import hashlib
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

import gevent
from gevent.lock import Semaphore
from locust import HttpUser, events, task

from bench.dataset import PromptProvider
from bench.metrics import MetricsCollector, RequestMetric, format_benchmark_report


metrics = MetricsCollector()
# 在 test_start 阶段初始化：为每次用户任务提供采样提示词。
prompt_provider: Optional[PromptProvider] = None
# 解析后的模型 ID：会写入每个请求的 payload。
resolved_model: str = ""
# 严格模式下的 tokenizer：用于精确控制 token 长度。
resolved_tokenizer = None
# 运行期健康标记：只要 CPU >= 90%，就记为客户端瓶颈。
cpu_bottleneck_detected = False
_cpu_monitor_stop = threading.Event()
_cpu_monitor_thread: Optional[threading.Thread] = None


class RequestRateLimiter:
    """Process-wide request start rate limiter.

    Locust's -r option controls user spawn rate, not HTTP request rate.
    This limiter spaces actual HTTP request starts according to --request-rate.
    """

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
        # 后台采样线程：压测期间持续监控本机 CPU。
        global cpu_bottleneck_detected
        while not _cpu_monitor_stop.is_set():
            usage = psutil.cpu_percent(interval=1.0)
            if usage >= 90.0:
                cpu_bottleneck_detected = True

    _cpu_monitor_thread = threading.Thread(target=_run, daemon=True)
    _cpu_monitor_thread.start()


def _stop_cpu_monitor() -> None:
    _cpu_monitor_stop.set()


def _resolve_model_id(host: str) -> str:
    # 未传 --model 时，从 OpenAI 兼容接口自动发现模型 ID。
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
        if isinstance(models, list) and models:
            first = models[0]
            if isinstance(first, dict):
                return str(first.get("id", "") or "")
    except Exception:
        return ""
    return ""


@events.init_command_line_parser.add_listener
def _(parser):
    # 在 Locust 启动前注册自定义命令行参数。
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


@events.test_start.add_listener
def _(environment, **kwargs):
    # 压测生命周期入口：初始化提示词来源、模型、tokenizer。
    global prompt_provider, resolved_model, resolved_tokenizer
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

        # 严格模式：只读取本地 tokenizer，避免线上自动拉取导致不一致。
        resolved_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"[bench] tokenizer loaded from: {tokenizer_source}")
    except Exception as exc:
        # 严格模式兜底：不依赖模型配置，直接加载 tokenizer.json。
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
    # 压测生命周期结束：停止监控并落盘汇总结果。
    opts = environment.parsed_options
    _stop_cpu_monitor()
    labeled_concurrency = int(getattr(opts, "benchmark_concurrency", 0) or 0)
    if labeled_concurrency <= 0:
        labeled_concurrency = int(getattr(opts, "num_users", 0) or 0)
    if labeled_concurrency <= 0 and environment.runner:
        labeled_concurrency = int(getattr(environment.runner, "target_user_count", 0) or 0)
    if labeled_concurrency <= 0 and environment.runner:
        labeled_concurrency = int(getattr(environment.runner, "user_count", 0) or 0)
    try:
        summary_data = metrics.summary(
            concurrency=labeled_concurrency,
            request_rate=opts.request_rate,
        )
        summary_data.update(
            {
                "host": environment.host,
                "model": opts.model,
                "client_cpu_bottleneck": str(cpu_bottleneck_detected).lower(),
            }
        )
        result_file = metrics.write_summary_csv(
            output_path=opts.summary_csv,
            summary_data=summary_data,
        )
        print(f"[bench] summary written: {result_file}")
        print(format_benchmark_report(summary_data))
        if cpu_bottleneck_detected:
            print("[bench] warning: client CPU >= 90% detected during this run")
    except PermissionError as exc:
        print(
            f"[bench] warning: failed to write summary csv ({exc}). "
            "Please close file handle (Excel/WPS) and rerun."
        )


class TRTLLMUser(HttpUser):
    # 无思考时间。真实请求启动速率由 RequestRateLimiter 控制。
    wait_time = lambda self: 0  # noqa: E731

    @staticmethod
    def _ensure_tokenizer_available() -> None:
        # 严格整形相关函数都依赖 tokenizer。
        if resolved_tokenizer is None:
            raise RuntimeError(
                "Tokenizer is required for strict input/output shaping. "
                "Please provide a valid --tokenizer-path."
            )

    @staticmethod
    def _build_prompt_with_exact_tokens(base_prompt: str, target_input_tokens: int) -> tuple[str, int, int, int]:
        # 构造用户 prompt，使其 token 长度尽量达到 target_input_tokens。
        # 返回值：(解码文本, 回编码长度, 截取后长度, 扩展后总编码长度)。
        TRTLLMUser._ensure_tokenizer_available()
        if target_input_tokens <= 0:
            return "", 0, 0, 0
        base = (base_prompt or "").strip()
        if not base:
            base = "hello"

        text = base
        token_ids = resolved_tokenizer.encode(text, add_special_tokens=False)
        piece = "\n" + base
        piece_ids = resolved_tokenizer.encode(piece, add_special_tokens=False)
        if not piece_ids:
            piece = " " + base
            piece_ids = resolved_tokenizer.encode(piece, add_special_tokens=False)
        if not piece_ids:
            raise RuntimeError("Tokenizer produced empty piece ids for prompt expansion")

        while len(token_ids) < target_input_tokens:
            # 分段扩展 prompt，避免一次性拼接过大文本。
            remaining = target_input_tokens - len(token_ids)
            repeats = max(1, min(128, remaining // max(len(piece_ids), 1)))
            text += piece * repeats
            token_ids = resolved_tokenizer.encode(text, add_special_tokens=False)

        expanded = token_ids[:target_input_tokens]
        decoded = resolved_tokenizer.decode(expanded, skip_special_tokens=False)
        reencoded = resolved_tokenizer.encode(decoded, add_special_tokens=False)
        return decoded, len(reencoded), len(expanded), len(token_ids)

    @staticmethod
    def _truncate_prompt_to_budget(prompt: str, token_budget: int) -> tuple[str, int]:
        # 按 token 预算硬截断（不是按字符数），用于满足后端上限。
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
        # 统计完整 chat prompt 的 token 数（包含模板/角色开销）。
        TRTLLMUser._ensure_tokenizer_available()
        # chat 模型优先走更精确的模板计数路径。
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
        # 保守兜底：手工拼接并附加固定开销保护。
        joined = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
        ids = resolved_tokenizer.encode(joined, add_special_tokens=True)
        return len(ids) + 256

    @task
    def chat_completion_stream(self):
        # 每个虚拟用户会反复执行的核心请求路径。
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
            # 1) 采样目标输入输出规模，并构造接近目标输入 token 的 prompt。
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
                # 当前 prompt 规模可接受，进入预算校验阶段。
                break
            last_build_error = (
                "Tokenizer decode/encode collapsed prompt unexpectedly. "
                f"attempt={attempt}, target_input_tokens={target_input_tokens}, "
                f"expanded={expanded_len}, reencoded={user_content_tokens}, "
                f"required_min={min_required}, len_base_prompt={len(sample.prompt)}, "
                f"sample_hash={sample_hash}"
            )
            print(f"[dbg] shape_retry {last_build_error}")
        else:
            raise RuntimeError(
                f"{last_build_error} (retries_exhausted={max_build_retries})"
            )
        server_max_tokens = int(opts.server_max_tokens)
        # 2) 发请求前先计算本地安全预算。
        allowed_prompt_tokens = server_max_tokens - int(opts.prompt_token_reserve) - max_tokens

        prompt_tokens = user_content_tokens

        if allowed_prompt_tokens <= 0:
            # 预算无效：输出 token + 预留已超过后端上限。
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
        # 3) 若超预算，按 token 迭代裁剪，直到满足 chat 模板总预算。
        final_prompt_tokens = self._count_chat_prompt_tokens(messages)
        while final_prompt_tokens > allowed_prompt_tokens and user_content_tokens > 0:
            overflow = final_prompt_tokens - allowed_prompt_tokens
            trim_step = max(overflow + 16, int(user_content_tokens * 0.05), 32)
            next_user_budget = max(0, user_content_tokens - trim_step)
            safe_prompt, user_content_tokens = self._truncate_prompt_to_budget(
                safe_prompt,
                next_user_budget,
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": safe_prompt},
            ]
            final_prompt_tokens = self._count_chat_prompt_tokens(messages)
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
            # 严格整形约束：最终 prompt 仍需接近目标输入规模。
            raise RuntimeError(
                "Prompt too short for strict shaping. "
                f"target_input_tokens={target_input_tokens}, "
                f"final_prompt_tokens={final_prompt_tokens}, "
                f"required_min={min_prompt_threshold}"
            )
        if final_prompt_tokens > allowed_prompt_tokens:
            # 裁剪后仍超预算：本地拒绝并记录失败样本。
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

        # 4) 真正的请求速率控制点。等待结束后才开始记录 TTFT/E2E。
        request_rate_limiter.wait()

        start = time.perf_counter()
        first_token_at = None
        output_fragments = []
        chunks_count = 0
        # 记录在途请求计数，用于并发/吞吐相关统计。
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
                    # 非 200 计为失败请求，并保留完整上下文指标。
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
                    # 5) 解析 SSE 流，并记录首 token 到达时间。
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
                        delta = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            output_fragments.append(delta)
                            chunks_count += 1
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                except Exception as exc:
                    # 流解析异常按请求失败处理。
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

        # 6) 将流式结果转换为基准指标并写入指标收集器。
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


