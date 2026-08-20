#!/usr/bin/env python3
"""
A stand-in for vLLM, so the whole pipeline can be exercised without a GPU.

It does NOT simulate the model. It produces answers that are format-valid for
each item, plus a controllable rate of unparseable ones, so that batching,
flushing, log writing, composite assembly, missing-value handling and CSV
output can all be tested on a laptop. The only thing left untested by a dry run
is the model call itself.

Used by generate_outcomes.py --dry_run.
"""

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class _Completion:
    text: str


@dataclass
class _Output:
    outputs: List[_Completion]


class StubTokenizer:
    """Just enough tokenizer to build a chat-wrapped prompt."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs) -> str:
        return messages[0]["content"]


class StubSamplingParams:
    """Mirrors the vLLM SamplingParams fields the pipeline sets."""

    def __init__(self, temperature=1.0, top_p=1.0, max_tokens=4,
                 guided_decoding=None):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.guided_decoding = guided_decoding


class StubGuidedDecodingParams:
    def __init__(self, regex: str):
        self.regex = regex


class StubLLM:
    """
    Answers each prompt with a value drawn from its own decoding regex.

    `failure_rate` injects answers that parse_answer() must reject, so the
    missing-value path is exercised rather than assumed.
    """

    def __init__(self, seed: int = 0, failure_rate: float = 0.01):
        self.rng = random.Random(seed)
        self.failure_rate = failure_rate

    def get_tokenizer(self) -> StubTokenizer:
        return StubTokenizer()

    def _answer_for(self, regex: Optional[str]) -> str:
        if self.rng.random() < self.failure_rate:
            return self.rng.choice(["", "I'd rather not say", "N/A", "999"])
        if regex is None:
            return "50"
        if "Yes" in regex:
            return self.rng.choice(["Yes", "No"])
        if regex == r"(10|[0-9])":
            return str(self.rng.randint(0, 10))
        return str(self.rng.randint(0, 100))

    def generate(self, prompts, sampling) -> List[_Output]:
        params = sampling if isinstance(sampling, list) else [sampling] * len(prompts)
        out = []
        for p in params:
            regex = p.guided_decoding.regex if p.guided_decoding else None
            out.append(_Output(outputs=[_Completion(self._answer_for(regex))]))
        return out
