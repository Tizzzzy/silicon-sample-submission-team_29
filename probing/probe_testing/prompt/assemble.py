#!/usr/bin/env python3
"""
Helper to reconstruct individual prompts from compact storage.

This module provides functions to iterate over the 1,584,000 (profile, item) pairs
and assemble each full prompt from the compact JSONL + JSON storage, ready to send
to an LLM.

Usage:
  from assemble import iter_all_prompts
  for profile_id, condition, item_key, submission_var, regex, max_tokens, full_prompt in iter_all_prompts():
      # Send full_prompt to LLM
      answer = llm_call(full_prompt, regex=regex, max_tokens=max_tokens)
      # Record answer
"""

import json
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple


def load_items_meta(items_meta_path: str = "items_meta.json") -> Dict[str, Dict]:
    """Load item metadata, keyed by item_key."""
    with open(items_meta_path) as f:
        items_list = json.load(f)
    return {item["item_key"]: item for item in items_list}


def load_prompts_jsonl(prompts_path: str = "prompts.jsonl") -> Generator[Dict, None, None]:
    """Yield one profile record at a time from the JSONL file."""
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def assemble_prompt(context: str, item_meta: Dict, extra_context_override: Optional[str] = None) -> str:
    """Reconstruct a full prompt from compact storage.

    Args:
        context: demographic preamble + transition + stimulus + final transition (from prompts.jsonl)
        item_meta: entry from items_meta (keyed by item_key)
        extra_context_override: override the item_meta's extra_context if needed

    Returns:
        Full prompt string, ready to send to LLM.
    """
    parts = [context]

    extra = extra_context_override if extra_context_override is not None else item_meta["extra_context"]
    if extra:
        parts.append(extra)

    parts.append(item_meta["question_block"])

    return "\n\n".join(parts)


def iter_all_prompts(
    prompts_path: str = "prompts.jsonl",
    items_meta_path: str = "items_meta.json",
) -> Generator[Tuple[str, str, str, str, str, int, str], None, None]:
    """Iterate over all 1,584,000 individual prompts.

    Yields tuples of:
        (profile_id, condition, item_key, submission_var, regex, max_tokens, full_prompt_text)

    Each tuple represents one (profile, item) pair, ready to send to an LLM.
    """
    items_meta = load_items_meta(items_meta_path)

    for profile_record in load_prompts_jsonl(prompts_path):
        profile_id = profile_record["profile_id"]
        condition = profile_record["condition"]
        context = profile_record["context"]

        for item_meta in items_meta.values():
            item_key = item_meta["item_key"]
            submission_var = item_meta["submission_var"]
            regex = item_meta["regex"]
            max_tokens = item_meta["max_tokens"]

            full_prompt = assemble_prompt(context, item_meta)

            yield (profile_id, condition, item_key, submission_var, regex, max_tokens, full_prompt)


def count_total_prompts(
    prompts_path: str = "prompts.jsonl",
    items_meta_path: str = "items_meta.json",
) -> int:
    """Return total number of prompts without iterating through all of them."""
    items_meta = load_items_meta(items_meta_path)
    n_items = len(items_meta)

    n_profiles = 0
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                n_profiles += 1

    return n_profiles * n_items


if __name__ == "__main__":
    # Example: print first 5 prompts to verify structure
    import sys
    count = 0
    for profile_id, condition, item_key, submission_var, regex, max_tokens, full_prompt in iter_all_prompts():
        count += 1
        if count <= 5:
            print(f"\n{'='*78}")
            print(f"Profile: {profile_id} | Condition: {condition}")
            print(f"Item: {item_key} | Submission var: {submission_var}")
            print(f"Regex: {regex} | Max tokens: {max_tokens}")
            print(f"Prompt preview (first 300 chars):\n{full_prompt[:300]}...")
        if count >= 5:
            break
    print(f"\n{'='*78}")
    print(f"Total prompts: {count_total_prompts():,}")
