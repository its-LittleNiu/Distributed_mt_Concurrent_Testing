# Distributed mt Concurrent Testing

这是一个基于 Locust 的 OpenAI 兼容接口压测项目，用于评估 TRT-LLM / PD 分离 / 大模型推理服务在不同并发和请求速率下的表现。

项目重点关注流式 Chat Completions 场景，输出请求吞吐、TTFT、TPOT、Token TPS、客户端 CPU 瓶颈等指标。

## 指标说明

每次压测会向 `results/summary.csv` 或指定的 `--summary-csv` 追加一行汇总结果，主要字段包括：

- `concurrency`：本轮标注并发数，矩阵脚本会通过 `--benchmark-concurrency` 显式传入。
- `request_rate`：单个 Locust 进程的请求启动速率。
- `input_tokens`：成功请求的输入 token 总数。
- `output_tokens`：成功请求的输出 token 总数。
- `success_count` / `fail_count`：成功和失败请求数。
- `avg_ttft` / `ttft_p50` / `ttft_p99`：首 token 延迟，单位为秒。
- `avg_tpot` / `tpot_p50` / `tpot_p99`：首 token 之后的平均输出 token 间隔，单位为秒。
- `tpot_sample_count` / `tpot_skipped_count`：参与 TPOT 统计和被跳过的样本数。输出 token 数小于等于 1 时，TPOT 不具备物理意义，会被跳过。
- `avg_latency` / `latency_p50` / `latency_p99`：端到端请求耗时，单位为秒。
- `output_tokens_per_s`：输出 token 吞吐。
- `total_tokens_per_s`：输入 + 输出 token 总吞吐。
- `request_throughput`：实际成功请求吞吐，单位 req/s。
- `peak_output_tokens_per_s`：按秒桶统计的峰值输出 token 吞吐。
- `peak_concurrent_requests`：观测到的峰值在途请求数。
- `avg_concurrency`：基于 Little's Law 估算的平均并发。
- `client_cpu_bottleneck`：客户端 CPU 是否达到 90% 以上。

## 目录结构

- `locustfile.py`：Locust 压测入口、请求构造、流式响应解析和指标采集。
- `bench/dataset.py`：ShareGPT 风格数据集读取和 prompt 采样。
- `bench/metrics.py`：指标收集、汇总、CSV 写入和报告格式化。
- `scripts/run_locust_matrix.py`：按并发和请求速率矩阵自动运行多轮压测。
- `scripts/format_benchmark_table.py`：从 Locust 日志提取汇总块，生成便于粘贴到表格软件的 TSV。
- `scripts/setup_tokenizer_only.py`：整理 tokenizer-only 目录。

## 安装依赖

```bash
pip install -r requirements.txt
```

如果需要记录客户端 CPU 瓶颈，请确保环境中安装了 `psutil`。未安装时程序会继续运行，但 `client_cpu_bottleneck` 标记不可用。

## 单轮压测

```bash
python -m locust -f locustfile.py \
  --headless \
  --host http://<SERVER_HOST>:<PORT> \
  -u 16 \
  -r 16 \
  --run-time 3m \
  --only-summary \
  --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --input-output "[4000:1000]" \
  --model <MODEL_ID> \
  --tokenizer-path ./tokenizer_only \
  --request-rate 1 \
  --summary-csv results/summary.csv \
  --server-max-tokens 8192 \
  --prompt-token-reserve 512 \
  --prompt-budget-ratio 1.0 \
  --seed 42
```

## 矩阵压测

常用矩阵压测命令：

```bash
mkdir -p logs results

LOG_FILE=logs/locust_4000x1000_workers4_$(date +"%Y%m%d_%H%M%S").log

python3 ./scripts/run_locust_matrix.py \
  --host http://<SERVER_HOST>:<PORT> \
  --concurrencies "16,32,64,96,128" \
  --request-rates "1,2,4" \
  --run-time 5m \
  --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --input-output "[4000:1000]" \
  --model <MODEL_ID> \
  --tokenizer-path ./tokenizer_only \
  --summary-csv results/summary_4000x1000_workers4_$(date +"%Y%m%d_%H%M%S").csv \
  --server-max-tokens 8192 \
  --prompt-token-reserve 512 \
  --prompt-budget-ratio 1.0 \
  --workers 4 \
  --seed 42 \
  2>&1 | tee "$LOG_FILE"
```

这个命令会运行：

- 5 个并发档位：`16,32,64,96,128`
- 3 个单进程请求速率档位：`1,2,4`
- 共 `5 x 3 = 15` 轮
- 每轮持续 `5m`

## request-rate 与 workers 的关系

`--request-rate` 是单个 Locust 进程的请求启动速率。

当使用：

```bash
--request-rates "1,2,4"
--workers 4
```

实际总请求启动速率约为：

| request_rate | workers | 总请求启动速率 |
|---:|---:|---:|
| 1 | 4 | 约 4 req/s |
| 2 | 4 | 约 8 req/s |
| 4 | 4 | 约 16 req/s |

这是多进程压测的预期行为。CSV 中的 `request_rate` 记录的是单进程速率，分析总压力时需要结合 workers 数量理解。

使用 `--request-rate 0` 或负数可以关闭请求启动速率限制，让 Locust 尽可能快地发起请求。

## Token 计数

项目启动时会加载 tokenizer：

```bash
--tokenizer-path ./tokenizer_only
```

输出 token 统计优先使用 tokenizer：

```python
tokenizer.encode(text, add_special_tokens=False)
```

如果未设置 tokenizer，才会回退到空格分词近似计数。对于中文、思维链和无空格文本，必须使用 tokenizer，否则 `output_tokens_per_s` 和 `total_tokens_per_s` 会明显失真。

Prompt 构造、截断和最终 prompt token 计数也依赖同一个 tokenizer。

## TTFT、TPOT 和延迟口径

- TTFT 从 HTTP POST 真正发出后开始计时，到收到第一个流式 token 为止。
- `request_rate_limiter.wait()` 的客户端限流等待不计入 TTFT 和 E2E latency。
- E2E latency 从 HTTP POST 发出到流式响应结束。
- TPOT 只在 `output_tokens > 1` 时计算；只有一个输出 token 时会跳过 TPOT 统计。

如果要进一步拆分服务端排队、Prefill 计算和 Decode 计算耗时，需要服务端侧 trace、日志或 timing header；客户端只能观测整体请求延迟。

## 结果表格格式化

如果需要把日志转换成这种表格：

```text
input-output    BS (conc) rate=1    TTFT (s)    TPOT (s)    RPS    TPS (tok/s)    Output TPS
[4000:1000]     16                  0.4878      0.0414      0.66   2991.01        352.76
```

可以使用：

```bash
python3 scripts/format_benchmark_table.py \
  logs/locust_4000x1000_workers4_xxx.log \
  --output results/benchmark_table_4000x1000_workers4.tsv
```

生成的 `.tsv` 可以直接用 Excel、WPS 或其他表格软件打开。

## 本机多进程与多机分布式

本机多进程推荐通过矩阵脚本：

```bash
python3 scripts/run_locust_matrix.py --workers 4
```

这会转成 Locust 的：

```bash
--processes 4
```

多机分布式时，Master 示例：

```bash
locust -f locustfile.py --master --host http://<SERVER_HOST>:<PORT>
```

Worker 示例：

```bash
locust -f locustfile.py --worker --master-host <MASTER_HOST>
```

## 注意事项

- `--model` 建议显式传入，避免服务端模型自动发现失败。
- `--seed` 控制数据集采样随机性，默认是 `42`。
- `--prompt-budget-ratio` 会按比例缩小目标输入 token，设置为 `1.0` 表示不额外缩小。
- `--server-max-tokens`、`--prompt-token-reserve` 和 `max_tokens` 共同决定 prompt 可用预算。
- 如果 CSV 已存在且字段发生变化，建议换一个新的 `--summary-csv` 文件名，避免旧表头和新字段不一致。
- 如果日志中出现 worker 汇总缺失警告，本轮 CSV 可能是部分结果，需要谨慎使用。
