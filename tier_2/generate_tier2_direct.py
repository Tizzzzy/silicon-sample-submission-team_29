#!/usr/bin/env python3
"""
Generate Tier-2 direct group-level predictions for the Silicon Sample Benchmark.

This script directly prompts the LLM to estimate group averages for each (condition)
and (condition × moderator × level) cell, rather than aggregating individual profiles.

This is an independent secondary-entry method (vs. the primary Tier-1 individual simulation).
Generates ~476 LLM calls (17 main cells + 459 moderator cells), significantly cheaper than
Tier-1's 9,000 individual respondents.

Key differences from Tier-1 prompting:
  - Asks the model to estimate the AVERAGE group response, not individual responses
  - Frames the task as "imagine you surveyed many people in this group"
  - For newsletter_signup, explicitly asks for a PROPORTION (0-1), not a yes/no choice
  - Temperature=0.0 by default (central-tendency estimation, not diverse sampling)

Usage:
  # Dry run (preview prompts, no GPU):
  python tier_2/generate_tier2_direct.py --dry_run

  # Smoke test (small subset):
  python tier_2/generate_tier2_direct.py --conditions_subset control,Consensus --max_retries 0

  # Full run:
  python tier_2/generate_tier2_direct.py --team_id team_29 --entry secondary-1 --version 1

Output:
  - predictions/team_29_T2_secondary-1_v1_cells_main.csv (221 rows)
  - predictions/team_29_T2_secondary-1_v1_cells_moderator.csv (5,967 rows)
  - raw_data_deposit/generate_tier2_direct_<timestamp>.jsonl (raw LLM outputs)
  - raw_data_deposit/generate_tier2_direct_<timestamp>.log (detailed logs)
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent dir to path for imports from LLM_simulation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "LLM_simulation"))

from generate_outcomes_qwen import (
    INTERVENTIONS,
    SCALE_RANGES,
    format_with_chat_template,
    parse_llm_output,
    compute_composite_outcomes,
    validate_outcomes,
    batched,
    setup_logging,
)
from questionnaire_parser import parse_intervention_texts


# ============================================================================
# CONSTANTS: Tier-2 Schema (from submission_spec.R)
# ============================================================================

TIER2_OUTCOMES = [
    "trust_multidimensional",
    "trust_post",
    "distrust_post",
    "funding_perceptions",
    "policy_role_mean",
    "inst_trust_mean",
    "belief_post",
    "concern_mean",
    "policy_general",
    "policy_specific_mean",
    "behavior_mean",
    "donation_ams",
    "newsletter_signup",
]

CONDITIONS = ["control"] + INTERVENTIONS  # 17 total

MODERATORS = {
    "gender": ["Male", "Female", "Other"],
    "age_band": ["18-29", "30-44", "45-59", "60+"],
    "race": [
        "White / Caucasian",
        "Black / African American",
        "Hispanic / Latino",
        "Asian / Asian American",
        "Other",
    ],
    "education": [
        "Less than high school",
        "High school diploma / GED",
        "Some college or Associate's degree",
        "Bachelor's degree",
        "Master's degree / Professional degree",
        "Doctorate degree / Ph.D.",
    ],
    "income": [
        "Less than $30,000",
        "$30,000 to $55,999",
        "$56,000 to $99,999",
        "$100,000 to $167,999",
        "$168,000 or more",
    ],
    "party": ["Republican", "Democrat", "Independent", "Other"],
}

# Sum of all moderator levels: 3+4+5+6+5+4 = 27
TOTAL_MODERATOR_LEVELS = sum(len(v) for v in MODERATORS.values())

MODERATOR_PHRASING = {
    "gender": "U.S. adults whose gender is {level}",
    "age_band": "U.S. adults in the {level} age group",
    "race": "U.S. adults whose race/ethnicity is {level}",
    "education": "U.S. adults whose highest level of education is {level}",
    "income": "U.S. adults whose household income is {level}",
    "party": "U.S. adults whose political party affiliation is {level}",
}


def get_all_items() -> List["Item"]:
    """
    Define all 43 survey items to ask about, one per prompt.

    Returns list of Item objects, each with question text and response variable name.
    """
    items = [
        # Competence (items 1-3)
        Item("trust_competence_1", 1,
             "How incompetent or competent are most climate scientists?",
             "Very incompetent", "Very competent"),
        Item("trust_competence_2", 2,
             "How unintelligent or intelligent are most climate scientists?",
             "Very unintelligent", "Very intelligent"),
        Item("trust_competence_3", 3,
             "How unqualified or qualified are most climate scientists?",
             "Very unqualified", "Very qualified"),
        # Integrity (items 4-6)
        Item("trust_integrity_1", 4,
             "How dishonest or honest are most climate scientists?",
             "Very dishonest", "Very honest"),
        Item("trust_integrity_2", 5,
             "How unethical or ethical are most climate scientists?",
             "Very unethical", "Very ethical"),
        Item("trust_integrity_3", 6,
             "How insincere or sincere are most climate scientists?",
             "Very insincere", "Very sincere"),
        # Benevolence (items 7-9)
        Item("trust_benevolence_1", 7,
             "How unconcerned or concerned are most climate scientists about people's wellbeing?",
             "Very unconcerned", "Very concerned"),
        Item("trust_benevolence_2", 8,
             "How uneager or eager are most climate scientists to improve others' lives?",
             "Very uneager", "Very eager"),
        Item("trust_benevolence_3", 9,
             "How inconsiderate or considerate are most climate scientists of others' interests?",
             "Very inconsiderate", "Very considerate"),
        # Openness (items 10-12)
        Item("trust_openness_1", 10,
             "How open, if at all, are most climate scientists to feedback?",
             "Not open at all", "Very open"),
        Item("trust_openness_2", 11,
             "How unwilling or willing are most climate scientists to be transparent?",
             "Very unwilling", "Very willing"),
        Item("trust_openness_3", 12,
             "How much or how little attention do climate scientists pay to other people's views?",
             "Very little attention", "A great deal of attention"),
        # Single-item trust & distrust (items 13-14)
        Item("trust_post", 13,
             "How much do you trust climate scientists?",
             "not at all", "very strongly"),
        Item("distrust_post", 14,
             "How much do you distrust climate scientists?",
             "not at all", "very strongly"),
        # Funding perceptions (item 15)
        Item("funding_perceptions", 15,
             "Do you think the federal government spends too much, too little, or about the right amount on climate change research?",
             "far too little", "far too much"),
        # Belief (item 16)
        Item("belief_post", 16,
             'How accurate do you think this statement is? "Human activities are causing climate change."',
             "not at all accurate", "extremely accurate"),
        # Policy role (items 17-20)
        Item("policy_role_1", 17,
             "Climate scientists should work closely with policy makers to integrate scientific results into policy-making.",
             "Strongly disagree", "Strongly agree"),
        Item("policy_role_2", 18,
             "Climate scientists should actively advocate for specific policies.",
             "Strongly disagree", "Strongly agree"),
        Item("policy_role_3", 19,
             "Climate scientists should communicate their findings to policy makers.",
             "Strongly disagree", "Strongly agree"),
        Item("policy_role_4", 20,
             "Climate scientists should be more involved in the policy-making process.",
             "Strongly disagree", "Strongly agree"),
        # Institutional trust (items 21-25)
        Item("inst_trust_epa", 21,
             "How much do you trust the EPA (Environmental Protection Agency)?",
             "not at all", "very strongly"),
        Item("inst_trust_nasa", 22,
             "How much do you trust NASA?",
             "not at all", "very strongly"),
        Item("inst_trust_noaa", 23,
             "How much do you trust NOAA?",
             "not at all", "very strongly"),
        Item("inst_trust_universities", 24,
             "How much do you trust Universities and colleges?",
             "not at all", "very strongly"),
        Item("inst_trust_federal_gov", 25,
             "How much do you trust the Federal government?",
             "not at all", "very strongly"),
        # Concern (items 26-28)
        Item("concern_1", 26,
             "How concerned are you about climate change?",
             "Not at all", "Extremely"),
        Item("concern_2", 27,
             "How serious a problem is climate change?",
             "Not at all", "Extremely"),
        Item("concern_3", 28,
             "Relative to other issues facing the U.S., how important is climate change?",
             "The least important issue", "The most important issue"),
        # Policy general (item 29)
        Item("policy_general", 29,
             'How much do you oppose or support: "The U.S. government should do more to reduce global warming"?',
             "Strongly oppose", "Strongly support"),
        # Policy specific (items 30-36)
        Item("policy_specific_1", 30,
             "Raising taxes on fossil fuels (e.g., gas, oil, coal)",
             "Strongly oppose", "Strongly support"),
        Item("policy_specific_2", 31,
             "Expanding infrastructure for public transportation",
             "Strongly oppose", "Strongly support"),
        Item("policy_specific_3", 32,
             "Increasing the use of sustainable energy such as wind and solar energy",
             "Strongly oppose", "Strongly support"),
        Item("policy_specific_4", 33,
             "Protecting forested and land areas",
             "Strongly oppose", "Strongly support"),
        Item("policy_specific_5", 34,
             "Increasing taxes on carbon-intensive foods (e.g., beef and dairy products)",
             "Strongly oppose", "Strongly support"),
        Item("policy_specific_6", 35,
             "Investing more in green jobs and businesses",
             "Strongly oppose", "Strongly support"),
        Item("policy_specific_7", 36,
             "Introducing laws to keep waterways and oceans clean",
             "Strongly oppose", "Strongly support"),
        # Behaviors (items 37-42)
        Item("behavior_meat", 37,
             "Eat less meat",
             "Not likely at all", "Extremely likely"),
        Item("behavior_transport", 38,
             "Walk, bicycle, carpool, or take public transportation more often instead of driving by yourself",
             "Not likely at all", "Extremely likely"),
        Item("behavior_solar", 39,
             "Install a solar panel",
             "Not likely at all", "Extremely likely"),
        Item("behavior_fly", 40,
             "Go on less personal (non-business) air travel",
             "Not likely at all", "Extremely likely"),
        Item("behavior_talk", 41,
             "Talk to friends and family about the importance of climate change",
             "Not likely at all", "Extremely likely"),
        Item("behavior_donate", 42,
             "Donate to an environmental NGO",
             "Not likely at all", "Extremely likely"),
        # Donation (item 43)
        Item("donation_ams", 43,
             "Of a $10 bonus, what is the AVERAGE amount this group would donate to the American Meteorological Society (AMS)?",
             "$0", "$10", scale_range=(0, 10)),
        # Newsletter (item 44) - kept for compatibility, though handled specially
        Item("newsletter_signup", 44,
             'What PROPORTION (share) of this group would subscribe to the "Talking Climate" newsletter?',
             "0 (nobody)", "1.0 (everyone)", scale_range=(0, 1)),
    ]
    return items


# ============================================================================
# CELL DEFINITION
# ============================================================================

@dataclass
class Cell:
    """Represents one (condition, moderator, level) cell to prompt."""

    condition: str
    moderator: Optional[str] = None  # None for main cells
    level: Optional[str] = None  # None for main cells

    @property
    def is_main(self) -> bool:
        """True for condition-only cells (no moderator)."""
        return self.moderator is None

    def __str__(self) -> str:
        if self.is_main:
            return f"{self.condition} [main]"
        return f"{self.condition} / {self.moderator} / {self.level}"


@dataclass
class Item:
    """Represents one survey item/question."""

    name: str  # Response variable name (e.g., "trust_competence_1")
    question_num: int  # Survey question number (1-43)
    question_text: str  # The actual question to ask
    scale_label_low: str  # e.g., "Very incompetent"
    scale_label_high: str  # e.g., "Very competent"
    scale_range: Tuple[int, int] = (0, 100)  # (min, max)


def describe_population(moderator: Optional[str], level: Optional[str]) -> str:
    """Describe the population for a given cell."""
    if moderator is None:
        return "a broad, nationally representative sample of U.S. adults"
    return MODERATOR_PHRASING[moderator].format(level=level)


def enumerate_cells(conditions_subset: Optional[List[str]] = None) -> Tuple[List[Cell], List[Cell]]:
    """
    Enumerate all cells (condition × demographic) to prompt.

    Returns:
        (main_cells, moderator_cells) where main_cells has 17 entries (one per condition)
        and moderator_cells has 17 * 27 = 459 entries.
    """
    conds = conditions_subset or CONDITIONS

    main_cells = [Cell(c, moderator=None, level=None) for c in conds]

    mod_cells = [
        Cell(c, moderator=mod, level=level)
        for c in conds
        for mod, levels in MODERATORS.items()
        for level in levels
    ]

    return main_cells, mod_cells


def enumerate_cell_item_pairs(cells: List[Cell]) -> List[Tuple[Cell, Item]]:
    """
    Enumerate all (cell, item) pairs for generation.

    For each cell, create one pair for each of the 43 survey items.
    Total: len(cells) × 43 pairs

    Returns:
        List of (Cell, Item) tuples
    """
    items = get_all_items()
    pairs = [(cell, item) for cell in cells for item in items]
    return pairs


def construct_group_survey_prompt(
    condition: str,
    stimulus_text: str,
    item: "Item",
    moderator: Optional[str],
    level: Optional[str],
    logger: logging.Logger,
) -> str:
    """
    Construct a single-item survey prompt for a group-level cell.

    Asks the LLM to estimate the AVERAGE response for ONE survey item only.

    Args:
        condition: The intervention condition name
        stimulus_text: The intervention stimulus text
        item: The Item to ask about (with question text, scale range, etc.)
        moderator: Moderator attribute name (None for main cells)
        level: Moderator level value (None for main cells)
        logger: Logger instance

    Returns:
        The prompt text asking for one item.
    """
    population = describe_population(moderator, level)

    # Determine the scale label for this item
    scale_min, scale_max = item.scale_range
    scale_label = f"({scale_min}={item.scale_label_low}, {scale_max}={item.scale_label_high})"

    # Special instructions for specific item types
    special_instruction = ""
    if item.name == "newsletter_signup":
        special_instruction = "\nReport a decimal between 0 and 1 — for example, 0.35 would mean 35% of this group would subscribe."
    elif item.name == "donation_ams":
        special_instruction = "\nReport the dollar amount (0-10, may include decimals)."

    prompt = f"""You are simulating the AVERAGE (typical) survey response of {population}, after this group is exposed to the following message.

This group is now reading the following message about climate scientists:

{stimulus_text}

---

After reading this message, report your best estimate of the AVERAGE response for this group on the following survey item:

**Question:** {item.question_text}

**Scale:** {scale_label}{special_instruction}

Please respond with ONLY a number. Do not include any other text.

RESPONSE_FORMAT:
{item.name}: [number]
"""
    return prompt.strip()


def generate_batch(
    llm: "LLM",
    sampling: "SamplingParams",
    cell_item_pairs: List[Tuple[Cell, Item]],
    interventions_dict: Dict[str, str],
    logger: logging.Logger,
) -> Dict[int, Optional[Tuple[str, float]]]:
    """
    Generate responses for a batch of (cell, item) pairs.

    Returns:
        Dict mapping pair index to (item_name, value) tuple (or None if parsing failed).
    """
    prompts = []
    for cell, item in cell_item_pairs:
        stim = interventions_dict[cell.condition]
        text = construct_group_survey_prompt(cell.condition, stim, item, cell.moderator, cell.level, logger)
        formatted = format_with_chat_template(llm, [{"type": "text", "text": text}])
        prompts.append({"prompt": formatted})

    outputs = llm.generate(prompts, sampling)
    parsed = {}

    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text.strip() if output.outputs else ""
        item_name = cell_item_pairs[i][1].name

        # Try to extract a single number from the response
        try:
            # Look for a number in the response (handles various formats)
            import re
            match = re.search(r'[-+]?\d*\.?\d+', raw_text)
            if match:
                value = float(match.group())
                # Clip to scale range
                scale_min, scale_max = cell_item_pairs[i][1].scale_range
                value = max(scale_min, min(scale_max, value))
                parsed[i] = (item_name, value)
            else:
                logger.warning(f"  Could not extract number from response: {raw_text[:100]}")
                parsed[i] = None
        except (ValueError, AttributeError) as e:
            logger.warning(f"  Error parsing response for {item_name}: {e}")
            parsed[i] = None

    return parsed


def run_with_retry(
    llm: "LLM",
    sampling: "SamplingParams",
    cell_item_pairs: List[Tuple[Cell, Item]],
    interventions_dict: Dict[str, str],
    logger: logging.Logger,
    batch_size: int,
    max_retries: int,
) -> Dict[Tuple[Cell, Item], Optional[float]]:
    """
    Generate responses for all (cell, item) pairs with retry logic.

    Returns:
        Dict mapping (Cell, Item) to value (or None if unrecoverable).
    """
    results = {}
    pending = list(cell_item_pairs)

    for attempt in range(max_retries + 1):
        if not pending:
            break

        logger.info(f"Attempt {attempt + 1}/{max_retries + 1}: {len(pending)} pending (cell, item) pairs")

        still_pending = []
        for batch in batched(pending, batch_size):
            parsed = generate_batch(llm, sampling, batch, interventions_dict, logger)

            for i, (cell, item) in enumerate(batch):
                if parsed[i] is None:
                    logger.warning(f"  [attempt {attempt + 1}] parse failed for {cell} / {item.name}")
                    still_pending.append((cell, item))
                else:
                    item_name, value = parsed[i]
                    results[(cell, item)] = value

        pending = still_pending

    for cell, item in pending:
        results[(cell, item)] = None  # Exhausted retries

    return results


def main_pipeline(
    llm: "LLM",
    sampling: "SamplingParams",
    interventions_dict: Dict[str, str],
    logger: logging.Logger,
    batch_size: int,
    max_retries: int,
    conditions_subset: Optional[List[str]] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str, str, str], float]]:
    """
    Run the full generation pipeline with two phases: main cells, then moderator cells.

    For each cell, generates responses for all 43 items, then aggregates to composites.

    Returns:
        (main_means, mod_means) where:
          - main_means: Dict[(condition, outcome) → mean_value]
          - mod_means: Dict[(condition, moderator, level, outcome) → mean_value]
    """
    main_cells, mod_cells = enumerate_cells(conditions_subset)

    # Phase A: Main cells
    logger.info(f"Phase A: Generating {len(main_cells)} main cells × 43 items = {len(main_cells) * 43} prompts...")
    main_cell_item_pairs = enumerate_cell_item_pairs(main_cells)
    main_raw_values = run_with_retry(llm, sampling, main_cell_item_pairs, interventions_dict, logger, batch_size, max_retries)

    # Aggregate raw item values back to cells, then to composites
    main_cell_raw_dict = {}  # Dict[Cell → Dict[item_name → value]]
    for (cell, item), value in main_raw_values.items():
        if cell not in main_cell_raw_dict:
            main_cell_raw_dict[cell] = {}
        if value is not None:
            main_cell_raw_dict[cell][item.name] = value

    main_means = {}
    for cell in main_cells:
        if cell in main_cell_raw_dict:
            raw_outcomes = main_cell_raw_dict[cell]
            composite_outcomes = compute_composite_outcomes(raw_outcomes)
        else:
            composite_outcomes = {}

        # Ensure all required outcomes are present
        for outcome in TIER2_OUTCOMES:
            if outcome in composite_outcomes:
                main_means[(cell.condition, outcome)] = composite_outcomes[outcome]
            else:
                # Fallback to neutral default
                if outcome == "newsletter_signup":
                    default_val = 0.5
                elif outcome == "donation_ams":
                    default_val = 5.0
                else:
                    default_val = 50.0
                logger.error(
                    f"FALLBACK: main cell {cell.condition}/{outcome} "
                    f"missing; using neutral default {default_val}"
                )
                main_means[(cell.condition, outcome)] = default_val

    # Phase B: Moderator cells
    logger.info(f"Phase B: Generating {len(mod_cells)} moderator cells × 43 items = {len(mod_cells) * 43} prompts...")
    mod_cell_item_pairs = enumerate_cell_item_pairs(mod_cells)
    mod_raw_values = run_with_retry(llm, sampling, mod_cell_item_pairs, interventions_dict, logger, batch_size, max_retries)

    # Aggregate raw item values back to cells, then to composites
    mod_cell_raw_dict = {}  # Dict[Cell → Dict[item_name → value]]
    for (cell, item), value in mod_raw_values.items():
        if cell not in mod_cell_raw_dict:
            mod_cell_raw_dict[cell] = {}
        if value is not None:
            mod_cell_raw_dict[cell][item.name] = value

    mod_means = {}
    for cell in mod_cells:
        if cell in mod_cell_raw_dict:
            raw_outcomes = mod_cell_raw_dict[cell]
            composite_outcomes = compute_composite_outcomes(raw_outcomes)
        else:
            composite_outcomes = {}

        # Ensure all required outcomes are present
        for outcome in TIER2_OUTCOMES:
            if outcome in composite_outcomes:
                mod_means[(cell.condition, cell.moderator, cell.level, outcome)] = composite_outcomes[outcome]
            else:
                # Fallback: repeat main-cell mean (FAQ policy)
                fallback = main_means[(cell.condition, outcome)]
                logger.warning(
                    f"FALLBACK: {cell}/{outcome} missing; repeating main-cell mean {fallback}"
                )
                mod_means[(cell.condition, cell.moderator, cell.level, outcome)] = fallback

    return main_means, mod_means


def assemble_main_df(main_means: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    """Assemble and validate the main cells dataframe."""
    rows = [
        {"condition": c, "outcome": o, "mean": v}
        for (c, o), v in main_means.items()
    ]
    df = pd.DataFrame(rows)[["condition", "outcome", "mean"]]

    # Validation
    assert len(df) == len(CONDITIONS) * len(TIER2_OUTCOMES) == 221, \
        f"Expected 221 rows, got {len(df)}"
    assert df["mean"].isna().sum() == 0, f"Found {df['mean'].isna().sum()} NaN values in main file"
    assert not df.duplicated(subset=["condition", "outcome"]).any(), "Found duplicate rows"
    assert set(df["condition"]) == set(CONDITIONS), "Condition mismatch"
    assert set(df["outcome"]) == set(TIER2_OUTCOMES), "Outcome mismatch"

    # Range checks
    for outcome in TIER2_OUTCOMES:
        vals = df.loc[df["outcome"] == outcome, "mean"]
        hi = 1 if outcome == "newsletter_signup" else (10 if outcome == "donation_ams" else 100)
        assert vals.between(0, hi).all(), \
            f"{outcome}: values outside [0, {hi}]: {vals[~vals.between(0, hi)].tolist()}"

    return df


def assemble_mod_df(mod_means: Dict[Tuple[str, str, str, str], float]) -> pd.DataFrame:
    """Assemble and validate the moderator cells dataframe."""
    rows = [
        {"condition": c, "moderator": m, "moderator_level": lvl, "outcome": o, "mean": v}
        for (c, m, lvl, o), v in mod_means.items()
    ]
    df = pd.DataFrame(rows)[["condition", "moderator", "moderator_level", "outcome", "mean"]]

    # Validation
    expected_rows = len(CONDITIONS) * TOTAL_MODERATOR_LEVELS * len(TIER2_OUTCOMES)
    assert len(df) == expected_rows == 5967, f"Expected {expected_rows} rows, got {len(df)}"
    assert df["mean"].isna().sum() == 0, f"Found {df['mean'].isna().sum()} NaN values in moderator file"
    assert not df.duplicated(subset=["condition", "moderator", "moderator_level", "outcome"]).any(), \
        "Found duplicate rows"

    for moderator, levels in MODERATORS.items():
        got = set(df.loc[df["moderator"] == moderator, "moderator_level"].unique())
        assert got == set(levels), f"{moderator}: expected {levels}, got {got}"

    # Range checks (same as main)
    for outcome in TIER2_OUTCOMES:
        vals = df.loc[df["outcome"] == outcome, "mean"]
        hi = 1 if outcome == "newsletter_signup" else (10 if outcome == "donation_ams" else 100)
        assert vals.between(0, hi).all(), \
            f"{outcome}: values outside [0, {hi}]: {vals[~vals.between(0, hi)].tolist()}"

    return df


def write_outputs(
    main_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    predictions_dir: Path,
    team_id: str,
    entry: str,
    version: int,
    logger: logging.Logger,
) -> Tuple[Path, Path]:
    """Write the output CSVs and return their paths."""
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    main_path = predictions_dir / f"{team_id}_T2_{entry}_v{version}_cells_main.csv"
    mod_path = predictions_dir / f"{team_id}_T2_{entry}_v{version}_cells_moderator.csv"

    main_df.to_csv(main_path, index=False)
    mod_df.to_csv(mod_path, index=False)

    logger.info(f"Wrote {main_path} ({len(main_df)} rows)")
    logger.info(f"Wrote {mod_path} ({len(mod_df)} rows)")

    return main_path, mod_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Tier-2 direct group-level predictions for Silicon Sample Benchmark"
    )
    parser.add_argument(
        "--questionnaire_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "survey" / "questionnaire.txt"),
        help="Path to survey/questionnaire.txt",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/projects/p32143/cache/qwen36_27b",
        help="Path to Qwen model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for vLLM generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0=deterministic, 1=max randomness)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Max retries per cell on parse/validate failure",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Tensor parallelism for vLLM",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "predictions"),
        help="Output directory for predictions CSVs",
    )
    parser.add_argument(
        "--raw_output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "raw_data_deposit"),
        help="Output directory for raw outputs and logs",
    )
    parser.add_argument(
        "--team_id",
        type=str,
        default="team_29",
        help="Team ID for output file naming",
    )
    parser.add_argument(
        "--entry",
        type=str,
        default="secondary-1",
        help="Entry label (primary, secondary-1, etc.)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version number for output files",
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        default=None,
        help="Truncate to first N cells (for testing)",
    )
    parser.add_argument(
        "--conditions_subset",
        type=str,
        default=None,
        help="Comma-separated subset of conditions to run (e.g. 'control,Consensus')",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview prompts and exit (no GPU needed)",
    )

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(args.raw_output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"generate_tier2_direct_{timestamp}.log"
    logger = setup_logging(str(log_path))

    logger.info("=" * 80)
    logger.info("SILICON SAMPLE BENCHMARK: TIER-2 DIRECT GROUP-LEVEL GENERATION")
    logger.info("=" * 80)
    logger.info(f"Questionnaire: {args.questionnaire_path}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Temperature: {args.temperature} (deterministic for group-level)")
    logger.info(f"Team ID: {args.team_id}, Entry: {args.entry}, Version: {args.version}")

    # Parse interventions
    logger.info("Loading intervention texts...")
    interventions_dict = parse_intervention_texts(args.questionnaire_path)
    logger.info(f"  Loaded {len(interventions_dict)} conditions")

    # Enumerate cells
    conditions_subset = None
    if args.conditions_subset:
        conditions_subset = args.conditions_subset.split(",")
        logger.info(f"  Subset: {conditions_subset}")

    main_cells, mod_cells = enumerate_cells(conditions_subset)

    if args.max_cells:
        all_cells = main_cells + mod_cells
        all_cells = all_cells[:args.max_cells]
        main_cells = [c for c in all_cells if c.is_main]
        mod_cells = [c for c in all_cells if not c.is_main]
        logger.info(f"  Truncated to {len(main_cells)} main + {len(mod_cells)} moderator = {len(all_cells)} total")
    else:
        logger.info(f"  {len(main_cells)} main cells + {len(mod_cells)} moderator cells = {len(main_cells) + len(mod_cells)} total")

    # Dry run: preview prompts
    if args.dry_run:
        logger.info("DRY RUN: Printing example prompts (Consensus, single items)...")
        consensus_stim = interventions_dict["Consensus"]
        items = get_all_items()

        # Show two example items
        example_item1 = items[0]  # trust_competence_1
        example_item2 = items[-1]  # newsletter_signup

        main_prompt_1 = construct_group_survey_prompt(
            "Consensus", consensus_stim, example_item1, None, None, logger
        )
        logger.info("\n" + "=" * 80)
        logger.info(f"EXAMPLE 1: Main cell (Consensus, {example_item1.name})")
        logger.info("=" * 80)
        print(main_prompt_1)

        mod_prompt_1 = construct_group_survey_prompt(
            "Consensus", consensus_stim, example_item2, "party", "Democrat", logger
        )
        logger.info("\n" + "=" * 80)
        logger.info(f"EXAMPLE 2: Moderator cell (Consensus, party=Democrat, {example_item2.name})")
        logger.info("=" * 80)
        print(mod_prompt_1)

        logger.info("\n" + "=" * 80)
        logger.info(f"Dry run complete. Total: {len(items)} items per cell")
        logger.info(f"  Main cells: {len(main_cells)} × {len(items)} = {len(main_cells) * len(items)} prompts")
        logger.info(f"  Moderator cells: {len(mod_cells)} × {len(items)} = {len(mod_cells) * len(items)} prompts")
        logger.info(f"  Total: {(len(main_cells) + len(mod_cells)) * len(items)} prompts")
        logger.info("No GPU/LLM calls made.")
        return

    # Initialize vLLM (import here to avoid dependency issues for dry-run mode)
    from vllm import LLM, SamplingParams
    logger.info("Initializing vLLM engine...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
        generation_config="vllm",
    )
    logger.info("vLLM initialized")

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Run pipeline
    main_means, mod_means = main_pipeline(
        llm,
        sampling,
        interventions_dict,
        logger,
        args.batch_size,
        args.max_retries,
        conditions_subset,
    )

    # Assemble & validate
    logger.info("Assembling main dataframe...")
    main_df = assemble_main_df(main_means)

    logger.info("Assembling moderator dataframe...")
    mod_df = assemble_mod_df(mod_means)

    # Write outputs
    logger.info("Writing outputs...")
    main_path, mod_path = write_outputs(
        main_df,
        mod_df,
        args.predictions_dir,
        args.team_id,
        args.entry,
        args.version,
        logger,
    )

    logger.info("=" * 80)
    logger.info("✓ TIER-2 GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Main cells:     {main_path}")
    logger.info(f"Moderator cells: {mod_path}")
    logger.info(f"Log file:        {log_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Validate output: head -3 <files>, check row counts")
    logger.info("  2. Register in metadata.json (manual step)")
    logger.info("  3. Run: make manifest && make check")


if __name__ == "__main__":
    main()
