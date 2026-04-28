import argparse
import json
import shutil
from pathlib import Path


REQUIRED_BASE_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
]

OPTIONAL_CODE_FILES = [
    "configuration_deepseek.py",
    "modeling_deepseek.py",
]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def ensure_auto_map(config_path: Path) -> None:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    auto_map = data.get("auto_map", {})
    auto_map.setdefault("AutoConfig", "configuration_deepseek.DeepseekV3Config")
    auto_map.setdefault("AutoModel", "modeling_deepseek.DeepseekV3Model")
    auto_map.setdefault(
        "AutoModelForCausalLM", "modeling_deepseek.DeepseekV3ForCausalLM"
    )
    data["auto_map"] = auto_map
    config_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenizer_only directory")
    parser.add_argument(
        "--source-dir",
        default=".",
        help="Directory containing tokenizer/config files",
    )
    parser.add_argument(
        "--code-source-dir",
        default=".",
        help="Directory containing configuration_deepseek.py/modeling_deepseek.py",
    )
    parser.add_argument(
        "--target-dir",
        default="./tokenizer_only",
        help="Target tokenizer directory",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    code_source_dir = Path(args.code_source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in REQUIRED_BASE_FILES:
        ok = copy_if_exists(source_dir / name, target_dir / name)
        if not ok:
            missing.append(name)
    if missing:
        raise SystemExit(f"Missing required files in source-dir: {missing}")

    for name in OPTIONAL_CODE_FILES:
        copied = copy_if_exists(code_source_dir / name, target_dir / name)
        if copied:
            print(f"copied {name}")
        else:
            print(f"warning: {name} not found in {code_source_dir}")

    ensure_auto_map(target_dir / "config.json")
    print(f"tokenizer_only prepared at: {target_dir}")


if __name__ == "__main__":
    main()
