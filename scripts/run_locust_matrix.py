import argparse
import json
import subprocess
import sys
from pathlib import Path


def rate_label(rate: float) -> str:
    text = f"{rate:g}"
    return text.replace("-", "neg_").replace(".", "_")


def result_json_path(summary_csv: str, users: int, rate: float) -> Path:
    summary_path = Path(summary_csv)
    output_dir = summary_path.parent / f"{summary_path.stem}_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"users_{users}_rate_{rate_label(rate)}.json"


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    result_json: str,
    expect_workers_max_wait: int,
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
        "--result-json",
        result_json,
        "--only-summary",
        "--exit-code-on-error",
        "0",
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
        cmd.extend(
            [
                "--processes",
                str(workers),
                "--expect-workers",
                str(workers),
                "--expect-workers-max-wait",
                str(expect_workers_max_wait),
            ]
        )

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def parse_csv_numbers(raw: str, tp=float):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(tp(part))
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
        "--expect-workers-max-wait",
        type=int,
        default=120,
        help="Maximum seconds to wait for distributed workers before starting a run",
    )
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Exit non-zero if any matrix run is invalid or crashes",
    )
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

    invalid_runs = []
    crashed_runs = []
    valid_runs = 0

    for users in concurrencies:
        for rate in request_rates:
            result_path = result_json_path(args.summary_csv, users, rate)
            if result_path.exists():
                result_path.unlink()

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
                result_json=str(result_path),
                expect_workers_max_wait=args.expect_workers_max_wait,
            )

            result = load_result(result_path)
            if result and result.get("valid"):
                valid_runs += 1
                summary = result.get("summary", {})
                print(
                    f"[matrix] valid users={users}, rate={rate}, "
                    f"success={summary.get('success_count', 0)}, "
                    f"fail={summary.get('fail_count', 0)}, "
                    f"elapsed_s={summary.get('elapsed_s', 0):.2f}"
                )
            else:
                if result:
                    reasons = result.get("reasons", [])
                    invalid_runs.append(
                        {
                            "users": users,
                            "rate": rate,
                            "exit_code": code,
                            "reasons": reasons,
                        }
                    )
                    print(
                        f"[matrix] invalid users={users}, rate={rate}, "
                        f"exit_code={code}, reasons={' | '.join(reasons) or 'unknown'}"
                    )
                else:
                    crashed_runs.append(
                        {
                            "users": users,
                            "rate": rate,
                            "exit_code": code,
                        }
                    )
                    print(
                        f"[matrix] missing result users={users}, rate={rate}, exit_code={code}"
                    )

            if code != 0 and not result:
                print(f"Run failed for users={users}, rate={rate}, exit_code={code}")

    print(
        f"[matrix] completed valid={valid_runs}, invalid={len(invalid_runs)}, "
        f"crashed={len(crashed_runs)}"
    )

    if args.fail_on_invalid and (invalid_runs or crashed_runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
