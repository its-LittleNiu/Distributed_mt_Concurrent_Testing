import argparse
import subprocess
import sys
from pathlib import Path


def run_once(
    host: str,
    users: int,
    rate: float,
    run_time: str,
    dataset: str,
    input_output: str,
    model: str,
    tokenizer_path: str,
    summary_csv: str,
    workers: int,
    server_max_tokens: int,
    prompt_token_reserve: int,
    prompt_budget_ratio: float,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "locustfile.py",
        "--headless",
        "--host",
        host,
        "-u",
        str(users),
        "-r",
        str(users),
        "--run-time",
        run_time,
        "--request-rate",
        str(rate),
        "--input-output",
        input_output,
        "--benchmark-concurrency",
        str(users),
        "--summary-csv",
        summary_csv,
        "--only-summary",
        "--server-max-tokens",
        str(server_max_tokens),
        "--prompt-token-reserve",
        str(prompt_token_reserve),
        "--prompt-budget-ratio",
        str(prompt_budget_ratio),
    ]
    if dataset:
        cmd.extend(["--dataset", dataset])
    if model:
        cmd.extend(["--model", model])
    if tokenizer_path:
        cmd.extend(["--tokenizer-path", tokenizer_path])
    if workers > 0:
        # Local pseudo-distributed mode with separate worker processes.
        cmd.extend(["--processes", str(workers)])

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def parse_csv_numbers(raw: str, tp=float):
    values = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        values.append(tp(p))
    return values


def main():
    parser = argparse.ArgumentParser(description="Run locust matrix benchmark")
    parser.add_argument("--host", default="http://10.10.240.13:8000")
    parser.add_argument("--concurrencies", default="16,32,64,96,128")
    parser.add_argument("--request-rates", default="1")
    parser.add_argument("--run-time", default="5m")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--input-output", default="[4000:1000],[6000:1000]")
    parser.add_argument("--model", default="DeepSeek-R1-0528", help="DeepSeek-R1-0528")
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="HF tokenizer path or local tokenizer directory",
    )
    parser.add_argument("--summary-csv", default="results/summary.csv")
    parser.add_argument("--workers", type=int, default=0, help="locust --processes N")
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
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("Missing --model. Please provide model id to avoid HTTP 400.")

    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    concurrencies = parse_csv_numbers(args.concurrencies, int)
    request_rates = parse_csv_numbers(args.request_rates, float)

    for users in concurrencies:
        for rate in request_rates:
            code = run_once(
                host=args.host,
                users=users,
                rate=rate,
                run_time=args.run_time,
                dataset=args.dataset,
                input_output=args.input_output,
                model=args.model,
                tokenizer_path=args.tokenizer_path,
                summary_csv=args.summary_csv,
                workers=args.workers,
                server_max_tokens=args.server_max_tokens,
                prompt_token_reserve=args.prompt_token_reserve,
                prompt_budget_ratio=args.prompt_budget_ratio,
            )
            if code != 0:
                print(f"Run failed for users={users}, rate={rate}, exit_code={code}")


if __name__ == "__main__":
    main()
