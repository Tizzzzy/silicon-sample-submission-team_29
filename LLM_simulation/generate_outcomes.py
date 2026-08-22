#!/usr/bin/env python3
"""
Generate Tier-1 outcomes by asking ONE survey item per call, in the survey's own
response format, with decoding constrained so the answer is always valid.

Replaces generate_outcomes_qwen.py. The differences that matter:

  old                                      new
  ---------------------------------------  ------------------------------------
  all 44 items in one response             one item per call
  free-form numbers, regex-scraped         decoding constrained to a valid answer
  parse failures filled with 50            failures recorded, never imputed
  14 of 17 stimuli were placeholder text   real stimulus text for all 17
  funding_perceptions stored unreversed    reversed per codebook.csv
  raw log truncated to 500 chars           full answer logged per item

Items are asked in the survey's own response format: sliders as integers 0-100,
the donation in whole dollars, the newsletter as Yes/No. Respondent demographics
are conditioned in the QA format of Jahanparast, Hong & Chang (ICLR 2026), which
is the only element taken from that paper. See PROMPTING.md.

Usage:
  python generate_outcomes.py \
    --profile_pool ../synthetic_profiles/profiles_pool.csv \
    --model_path /projects/p32143/cache/qwen36_27b \
    --questionnaire ../survey/questionnaire.txt \
    --output_dir ../raw_data_deposit \
    --seed 2026
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add the script's directory to sys.path so imports work when run from elsewhere
sys.path.insert(0, str(Path(__file__).parent))

from items import BY_KEY, ITEMS, load_newsletter_offer
from prompt_qa import ItemPrompt, build_item_prompt, parse_answer
from stimuli import parse_stimuli, render_extreme_weather, stimulus_for

# Tier-1 submission columns, in order (submission_spec.R).
SUBMISSION_COLUMNS = [
    "profile_id", "condition",
    "gender", "age_band", "race", "education", "income", "party",
    # trust_multidimensional comes BEFORE the sub-items -- this is the order in
    # scripts/lib/submission_spec.R and the organizers' example file. The old
    # pipeline had it after the sub-items.
    "trust_multidimensional",
    "trust_competence_1", "trust_competence_2", "trust_competence_3",
    "trust_integrity_1", "trust_integrity_2", "trust_integrity_3",
    "trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3",
    "trust_openness_1", "trust_openness_2", "trust_openness_3",
    "trust_post", "distrust_post", "funding_perceptions",
    "policy_role_mean", "inst_trust_mean", "belief_post", "concern_mean",
    "policy_general", "policy_specific_mean", "behavior_mean",
    "donation_ams", "newsletter_signup",
]

log = logging.getLogger("generate_outcomes")


def setup_logging(log_file: Path) -> None:
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(console)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"))
    log.addHandler(handler)


def respondent_stimulus(profile: Dict, stimuli: Dict[str, str], rng: random.Random) -> str:
    """The exact text this respondent sees, including the two adaptive arms."""
    condition = profile["condition"]
    if condition == "Extreme weather predictions":
        return render_extreme_weather(stimuli[condition], profile.get("state"))
    return stimulus_for(condition, stimuli, rng)


def build_prompts_for_profile(
    profile: Dict, stimuli: Dict[str, str], newsletter_offer: str, style: str,
    rng: random.Random,
) -> List[ItemPrompt]:
    """44 prompts sharing one prefix, so prefix caching pays for itself."""
    stimulus = respondent_stimulus(profile, stimuli, rng)
    prompts = []
    for item in ITEMS:
        # The newsletter item asks about an offer page shown just before it.
        extra = newsletter_offer if item.key == "newsletter" else None
        prompts.append(build_item_prompt(
            profile, stimulus, item, style=style, extra_context=extra))
    return prompts


def composites(item_values: Dict[str, float]) -> Dict[str, float]:
    """
    Fold the 44 item values into the 25 submission columns.

    Items sharing a submission_var are averaged. funding_perceptions is
    reverse-coded (100 - raw) per codebook.csv. Returns only what is present:
    a missing item propagates to a missing column rather than a filled midpoint.
    """
    buckets: Dict[str, List[float]] = defaultdict(list)
    for item in ITEMS:
        if item.key not in item_values:
            continue
        value = item_values[item.key]
        if item.reverse:
            value = 100.0 - value
        buckets[item.submission_var].append(value)

    out = {var: sum(vals) / len(vals) for var, vals in buckets.items()}

    # Primary outcome: mean of the four trust subscales, each a mean of 3 items.
    subscales = []
    for scale in ("competence", "integrity", "benevolence", "openness"):
        parts = [out[f"trust_{scale}_{i}"] for i in (1, 2, 3)
                 if f"trust_{scale}_{i}" in out]
        if len(parts) == 3:
            subscales.append(sum(parts) / 3)
    if len(subscales) == 4:
        out["trust_multidimensional"] = sum(subscales) / 4

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile_pool", default="../synthetic_profiles/profiles_pool.csv")
    parser.add_argument("--questionnaire", default="../survey/questionnaire.txt")
    parser.add_argument("--model_path", default="/projects/p32143/cache/qwen36_27b")
    parser.add_argument("--output_dir", default="../raw_data_deposit")
    parser.add_argument("--style", default="qa", choices=["qa", "bio", "portray"],
                        help="Demographic conditioning format (ICLR paper Table 2)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="OPEN DECISION. This is the only source of variation "
                             "between two respondents with identical demographics "
                             "in the same condition. At 0 they get identical "
                             "answers; at 1 the spread is the model's own. See "
                             "CHANGELOG.md before changing.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Prompts per vLLM generate() call, not respondents")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_profiles", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run the whole pipeline with a stub model instead of "
                             "vLLM. No GPU needed. Answers are random, so output "
                             "is for testing the pipeline only.")
    parser.add_argument("--dry_run_failure_rate", type=float, default=0.01,
                        help="Share of stub answers that are deliberately "
                             "unparseable, to exercise the missing-value path")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / f"generate_outcomes_{timestamp}.log")

    log.info("=" * 78)
    log.info("SILICON SAMPLE BENCHMARK — one item per call")
    log.info("=" * 78)
    for k, v in vars(args).items():
        log.info(f"  {k}: {v}")

    stimuli = parse_stimuli(args.questionnaire)
    log.info(f"Parsed {len(stimuli)} stimulus blocks from {args.questionnaire}")
    newsletter_offer = load_newsletter_offer(args.questionnaire)

    profiles = pd.read_csv(args.profile_pool)
    if args.max_profiles:
        profiles = profiles.head(args.max_profiles)
    log.info(f"Loaded {len(profiles):,} profiles → {len(profiles) * len(ITEMS):,} calls")

    if args.dry_run:
        # No GPU, no model. Exercises everything except the model call itself.
        from stub_llm import StubGuidedDecodingParams as GuidedDecodingParams
        from stub_llm import StubLLM, StubSamplingParams as SamplingParams
        log.warning("DRY RUN: answers are random, not model output. "
                    "Use this to test the pipeline, never to produce a submission.")
        llm = StubLLM(seed=args.seed, failure_rate=args.dry_run_failure_rate)
    else:
        # Import late so --help and --dry_run work without a GPU present.
        from vllm import LLM, SamplingParams
        try:
            from vllm.sampling_params import GuidedDecodingParams
        except ImportError:  # older vLLM
            from vllm.model_executor.guided_decoding.guided_fields import (  # type: ignore
                GuidedDecodingRequest as GuidedDecodingParams)
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,  # the 44 items of a respondent share a prefix
            trust_remote_code=True,
            disable_log_stats=True,
            seed=args.seed,
        )
    tokenizer = llm.get_tokenizer()

    # Decoding is constrained per item, so the model cannot answer out of range
    # or in the wrong format. Temperature is what produces variation between two
    # respondents with identical demographics in the same condition -- at 0 they
    # would receive identical answers. See CHANGELOG.md.
    def chat_wrap(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def sampling_for(prompt: ItemPrompt):
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=prompt.max_tokens,
            guided_decoding=GuidedDecodingParams(regex=prompt.regex),
        )

    rng = random.Random(args.seed)
    raw_path = output_dir / f"raw_output_{timestamp}.jsonl"
    rows: List[Dict] = []
    n_missing = 0

    with open(raw_path, "w") as raw_file:
        pending: List[tuple] = []   # (profile_dict, ItemPrompt)

        def flush() -> None:
            nonlocal pending, n_missing
            if not pending:
                return
            outputs = llm.generate(
                [{"prompt": chat_wrap(p.text)} for _, p in pending],
                [sampling_for(p) for _, p in pending],
            )

            by_profile: Dict[str, Dict[str, float]] = defaultdict(dict)
            profile_of: Dict[str, Dict] = {}

            for (profile, prompt), output in zip(pending, outputs):
                pid = profile["profile_id"]
                profile_of[pid] = profile
                text = output.outputs[0].text if output.outputs else ""
                item = BY_KEY[prompt.item_key]
                value = parse_answer(text, item)

                record = {
                    "profile_id": pid,
                    "condition": profile["condition"],
                    "item": prompt.item_key,
                    "raw_answer": text,
                    "value": value,
                }
                if value is None:
                    # Recorded as missing, never imputed. The old pipeline wrote
                    # 50 here, indistinguishable from a real neutral answer.
                    n_missing += 1
                    record["error"] = "unparseable answer"
                    log.warning(f"{pid}/{prompt.item_key}: unparseable answer {text!r}")
                raw_file.write(json.dumps(record) + "\n")
                if value is not None:
                    by_profile[pid][prompt.item_key] = value

            for pid, item_values in by_profile.items():
                profile = profile_of[pid]
                row = {c: profile[c] for c in SUBMISSION_COLUMNS[:8]}
                row.update(composites(item_values))
                rows.append(row)

            pending = []

        for _, profile_row in profiles.iterrows():
            profile = profile_row.to_dict()
            for prompt in build_prompts_for_profile(
                    profile, stimuli, newsletter_offer, args.style, rng):
                pending.append((profile, prompt))
            # Flush on respondent boundaries so a profile's 44 items stay together.
            if len(pending) >= args.batch_size:
                flush()
        flush()

    log.info(f"Raw per-item answers written to {raw_path}")
    if n_missing:
        log.warning(f"{n_missing:,} item calls produced no usable answer "
                    f"({100 * n_missing / (len(profiles) * len(ITEMS)):.3f}% of calls)")

    df = pd.DataFrame(rows)
    missing_cols = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
    if missing_cols:
        log.error(f"Submission columns never produced a value: {missing_cols}")
    df = df.reindex(columns=SUBMISSION_COLUMNS)

    out_csv = output_dir / f"tier1_submission_{timestamp}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv} ({len(df):,} rows x {len(df.columns)} cols)")

    n_incomplete = int(df.isna().any(axis=1).sum())
    if n_incomplete:
        log.warning(f"{n_incomplete:,} rows have at least one missing outcome. "
                    f"Decide how to handle these before submitting; they are NOT "
                    f"filled in automatically.")

    log.info("-" * 78)
    for var in ("trust_multidimensional", "trust_post", "funding_perceptions",
                "donation_ams", "newsletter_signup"):
        if var in df.columns:
            col = df[var]
            log.info(f"{var:26s} mean={col.mean():6.2f} sd={col.std():5.2f} "
                     f"min={col.min():6.2f} max={col.max():6.2f} n_missing={col.isna().sum()}")
    log.info("-" * 78)
    log.info("Next: scripts/clean.R (or write predictions/ directly), then make check")





if __name__ == "__main__":
    main()
