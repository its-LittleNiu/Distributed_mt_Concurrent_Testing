# 基于 Locust 的 TRT-LLM 分布式并发压测工具

本项目提供一套偏企业交付场景的压测基线。主要用于测试 TRT-LLM OpenAI 兼容服务，尤其适合 Prefill/Decode 分离部署后的接口压测。

## 测试指标

每次运行后，脚本会将以下 KPI 字段写入 `results/summary.csv`：

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

## 项目结构

- `locustfile.py`：定义 Locust 用户行为，并采集流式响应指标。
- `bench/dataset.py`：加载 ShareGPT 类数据集，并按目标输入长度构造 prompt。
- `bench/metrics.py`：内存指标收集器，并负责生成 CSV 汇总文件。
- `scripts/run_locust_matrix.py`：自动执行并发数和请求速率矩阵测试。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

单轮压测：

```bash
python -m locust -f locustfile.py --headless --host http://host:8000 -u 16 -r 16 --run-time 3m --only-summary --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --input-output "[4000:1000],[6000:1000]" --request-rate 2 --summary-csv results/summary.csv
```

如果服务端需要显式指定模型 ID，可以增加：

```bash
--model <your_model_id>
```

为了更准确地裁剪 prompt，建议增加 tokenizer 路径：

```bash
--tokenizer-path <hf_repo_or_local_path>
```

矩阵压测，形式接近 SGLang benchmark 的使用方式：

```bash
python scripts/run_locust_matrix.py --host http://host:8000 --concurrencies "16,32,64,96,128" --request-rates "1" --run-time 5m --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --input-output "[4000:1000],[6000:1000]" --model <your_model_id> --tokenizer-path <hf_repo_or_local_path> --summary-csv results/summary.csv
```

对于 8k 上下文限制的模型，可以增加：

```bash
--server-max-tokens 8192 --prompt-token-reserve 128 --prompt-budget-ratio 0.6
```

## 请求速率控制

`--request-rate` 是真实的 HTTP 请求启动速率限制器。

它不是 summary 里的一个展示字段。

例如：

```bash
--request-rate 4
```

这表示单个 Locust 进程大约每秒启动 4 个 HTTP 请求。

限速逻辑会在 `POST /v1/chat/completions` 之前立即生效。

prompt 构造、本地 prompt 裁剪、本地拒绝等步骤，不会计入请求延迟。

因此，TTFT 和 E2E latency 仍然从真实 HTTP POST 发起时开始计算。

可以用下面的方式做分层压力测试：

```bash
python scripts/run_locust_matrix.py \
  --host http://host:8000 \
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

使用 `--request-rate 0` 或负数，可以关闭请求速率限制。此时 Locust 会尽可能快地发送请求。

注意：使用 `--processes N` 时，每个 Locust 进程都会独立应用自己的 `--request-rate`。因此总请求启动速率约等于 `request_rate * N`。

## 真正的分布式模式，多机器压测

Master 节点：

```bash
locust -f locustfile.py --master --host http://host:8000
```

Worker 节点：

```bash
locust -f locustfile.py --worker --master-host <master_ip>
```

如果只想在一台机器上模拟 CI 类似的多进程分布式压测，可以使用：

```bash
python scripts/run_locust_matrix.py --workers 4
```

该参数会把 `--processes 4` 传递给 Locust。

## 说明

- 默认 token 统计采用空格切分，作为模型无关的基线估算方式。
- 如果没有提供数据集路径，benchmark 会回退到内置 prompt。
- prompt 会优先使用模型 tokenizer 进行裁剪。如果 tokenizer 加载失败，则回退到按词切分。裁剪预算为：
  `(server_max_tokens - max_tokens - prompt_token_reserve) * prompt_budget_ratio`。
- 如果某轮测试中客户端 CPU 使用率达到 90% 以上，summary 行中会写入 `client_cpu_bottleneck=true`。
- 如果需要严格的 tokenizer 级 token 统计，可以将 `bench/metrics.py` 中的 `estimate_tokens()` 替换为生产环境使用的 tokenizer。
