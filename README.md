# Locust Distributed Concurrent Benchmark for TRT-LLM

This project provides an enterprise-style load test baseline for disaggregated TRT-LLM OpenAI-compatible serving.

## What it measures

For each run, the script exports these KPI columns into `results/summary.csv`:

- `concurrency`
- `request_rate`
- `input_tokens`
- `output_tokens`
- `success_count`
- `fail_count`
- `avg_ttft`
- `avg_latency`
- `output_tokens_per_s`
- `total_tokens_per_s`
- `client_cpu_bottleneck`

## Project layout

- `locustfile.py`: Locust user behavior and stream metrics collection
- `bench/dataset.py`: ShareGPT-like dataset loader and prompt shaper
- `bench/metrics.py`: In-memory collector and CSV summary writer
- `scripts/run_locust_matrix.py`: run concurrency/rate matrix automatically

## Quick start

Install:

```bash
pip install -r requirements.txt
```

Single run:

```bash
python -m locust -f locustfile.py --headless --host http://10.10.240.13:8000 -u 16 -r 16 --run-time 3m --only-summary --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --input-output "[4000:1000],[6000:1000]" --request-rate 2 --summary-csv results/summary.csv
```

If your server requires explicit model id (common), add:

```bash
--model <your_model_id>
```

For accurate prompt clipping, add tokenizer source (recommended):

```bash
--tokenizer-path <hf_repo_or_local_path>
```

Matrix run (similar to your SGLang benchmark style):

```bash
python scripts/run_locust_matrix.py --host http://10.10.240.13:8000 --concurrencies "16,32,64,96,128" --request-rates "1" --run-time 5m --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --input-output "[4000:1000],[6000:1000]" --model <your_model_id> --tokenizer-path <hf_repo_or_local_path> --summary-csv results/summary.csv
```

For models with 8k context limits, add:

```bash
--server-max-tokens 8192 --prompt-token-reserve 128 --prompt-budget-ratio 0.6
```

## Request rate control

`--request-rate` is an actual HTTP request start-rate limiter.

It is not only a display field in the summary.

For example:

```bash
--request-rate 4
```

This means one Locust process starts about 4 HTTP requests per second.

The limiter is applied immediately before `POST /v1/chat/completions`.

Prompt construction, local prompt clipping, and local rejection are not counted as request latency.

Therefore, TTFT and E2E latency still start from the real HTTP POST time.

Use this pattern to test layered pressure:

```bash
python scripts/run_locust_matrix.py \
  --host http://10.10.240.13:8000 \
  --concurrencies "16,32,64,96,128" \
  --request-rates "1,2,4" \
  --run-time 5m \
  --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --input-output "[4000:1000],[6000:1000]" \
  --model DeepSeek-R1-0528 \
  --tokenizer-path ./tokenizer_only \
  --summary-csv results/summary.csv \
  --server-max-tokens 8192 \
  --prompt-token-reserve 512 \
  --prompt-budget-ratio 1.0
```

Use `--request-rate 0` or a negative value to disable request-rate limiting and let Locust send requests as fast as possible.

Note: when using `--processes N`, each Locust process applies its own `--request-rate`. The total request start rate is approximately `request_rate * N`.

## True distributed mode (multi-machine)

Master node:

```bash
locust -f locustfile.py --master --host http://10.10.240.13:8000
```

Worker node(s):

```bash
locust -f locustfile.py --worker --master-host <master_ip>
```

For CI-like local process distribution on one machine, use:

```bash
python scripts/run_locust_matrix.py --workers 4
```

This forwards `--processes 4` to locust.

## Notes

- Token count is estimated by whitespace split for a model-agnostic baseline.
- If dataset path is missing, benchmark falls back to built-in prompts.
- Prompt will be clipped by the model tokenizer first (fallback to word split if tokenizer load fails), using a conservative budget:
  `(server_max_tokens - max_tokens - prompt_token_reserve) * prompt_budget_ratio`.
- If client CPU usage reaches 90%+ during a run, `client_cpu_bottleneck=true` is written to the summary row.
- For strict tokenizer-level accounting, replace `estimate_tokens()` in `bench/metrics.py` with your production tokenizer.
