import argparse
import re
from pathlib import Path


FIELD_PATTERNS = {
    "request_rate": re.compile(r"Traffic request rate:\s*([0-9.]+)"),
    "concurrency": re.compile(r"Max request concurrency:\s*([0-9.]+)"),
    "ttft_ms": re.compile(r"Mean TTFT \(ms\):\s*([0-9.]+)"),
    "tpot_ms": re.compile(r"Mean TPOT \(ms\):\s*([0-9.]+)"),
    "rps": re.compile(r"Request throughput \(req/s\):\s*([0-9.]+)"),
    "tps": re.compile(r"Total token throughput \(tok/s\):\s*([0-9.]+)"),
    "output_tps": re.compile(r"Output token throughput \(tok/s\):\s*([0-9.]+)"),
}


def _format_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _format_rate(value: float) -> str:
    return str(int(value)) if value.is_integer() else _format_float(value, 4)


def _detect_input_output(lines: list[str], fallback: str) -> str:
    for line in lines:
        match = re.search(r"--input-output\s+(\S+)", line)
        if match:
            return match.group(1)
    return fallback


def parse_log(log_path: Path, input_output: str) -> list[dict]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    detected_input_output = _detect_input_output(lines, input_output)
    rows = []
    current = None

    for line in lines:
        if "Serving Benchmark Result" in line:
            current = {"input_output": detected_input_output}
            continue

        if current is None:
            continue

        for key, pattern in FIELD_PATTERNS.items():
            match = pattern.search(line)
            if match:
                current[key] = float(match.group(1))

        if line.startswith("===") and current:
            required = set(FIELD_PATTERNS)
            if required.issubset(current):
                rows.append(current)
            current = None

    return rows


def render_table(rows: list[dict]) -> str:
    grouped: dict[float, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["request_rate"], []).append(row)

    blocks = []
    for rate in sorted(grouped):
        header = [
            "input-output",
            f"BS (conc) rate={_format_rate(rate)}",
            "TTFT (s)",
            "TPOT (s)",
            "RPS",
            "TPS (tok/s)",
            "Output TPS",
        ]
        lines = ["\t".join(header)]
        for row in sorted(grouped[rate], key=lambda item: item["concurrency"]):
            lines.append(
                "\t".join(
                    [
                        row["input_output"],
                        str(int(row["concurrency"])),
                        _format_float(row["ttft_ms"] / 1000.0, 4),
                        _format_float(row["tpot_ms"] / 1000.0, 4),
                        _format_float(row["rps"], 2),
                        _format_float(row["tps"], 2),
                        _format_float(row["output_tps"], 2),
                    ]
                )
            )
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Format benchmark log into grouped TSV tables")
    parser.add_argument("log_file", help="Locust log file containing Serving Benchmark Result blocks")
    parser.add_argument(
        "--input-output",
        default="[4000:1000]",
        help="Fallback input-output label if it cannot be detected from the log",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output TSV path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    rows = parse_log(Path(args.log_file), args.input_output)
    if not rows:
        raise SystemExit(f"No benchmark summary blocks found in {args.log_file}")

    rendered = render_table(rows)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"formatted table written: {output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
