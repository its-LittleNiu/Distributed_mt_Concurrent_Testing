import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PromptSample:
    prompt: str
    target_input_tokens: int
    target_output_tokens: int


DEFAULT_PROMPTS = [
    PromptSample("Explain how transformers work in one paragraph.", 4000, 128),
    PromptSample("Write a Python function for quick sort.", 4000, 196),
    PromptSample("Summarize the benefits of distributed inference.", 6000, 160),
]


class PromptProvider:
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        input_output_pairs: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.random = random.Random(seed)
        self.prompts: List[str] = []
        self.length_pairs = self._parse_length_pairs(input_output_pairs)
        self._load(dataset_path)

    @staticmethod
    def _parse_length_pairs(input_output_pairs: Optional[str]) -> List[tuple[int, int]]:
        """
        Parse string like: [4000:1000],[6000:1000]
        """
        if not input_output_pairs:
            return []
        cleaned = input_output_pairs.replace(" ", "")
        parts = [p.strip("[]") for p in cleaned.split(",") if p.strip()]
        pairs: List[tuple[int, int]] = []
        for part in parts:
            if ":" not in part:
                continue
            in_len, out_len = part.split(":")
            if in_len.isdigit() and out_len.isdigit():
                pairs.append((int(in_len), int(out_len)))
        return pairs

    def _load(self, dataset_path: Optional[str]) -> None:
        if dataset_path and Path(dataset_path).exists():
            self.prompts = self._load_sharegpt_like(dataset_path)
        if not self.prompts:
            self.prompts = [s.prompt for s in DEFAULT_PROMPTS]
        if not self.length_pairs:
            self.length_pairs = [(4000, 1000)]

    def _load_sharegpt_like(self, dataset_path: str) -> List[str]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        loaded: List[str] = []
        for row in records:
            conversations = row.get("conversations", [])
            user_text = None
            for turn in conversations:
                if turn.get("from") == "human":
                    user_text = turn.get("value", "")
                    break
            if not user_text:
                continue
            loaded.append(user_text)
        return loaded

    def sample(self) -> PromptSample:
        prompt = self.random.choice(self.prompts)
        target_input_tokens, target_output_tokens = self.random.choice(self.length_pairs)
        return PromptSample(
            prompt=prompt,
            target_input_tokens=target_input_tokens,
            target_output_tokens=target_output_tokens,
        )
