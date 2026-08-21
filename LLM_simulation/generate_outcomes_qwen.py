#!/usr/bin/env python3
"""
Generate all 13 outcomes + 12 trust sub-items for synthetic respondent profiles
using the Qwen LLM and vLLM inference engine.

This script:
1. Loads the synthetic profile pool (profile_id, condition, 6 demographics)
2. For each profile, constructs a prompt with:
   - Demographic persona description
   - The assigned intervention condition/stimulus
   - Survey questions for all outcomes
3. Batches prompts and calls Qwen via vLLM
4. Parses LLM output to extract outcome values (0-100 scales, 0-10 donation, 0-1 newsletter)
5. Assembles final Tier-1 submission CSV with all 25 required columns

Outcome variables (13 + 12 trust sub-items):
  - trust_multidimensional (PRIMARY, mean of 4 subscales)
  - trust_competence_1, trust_competence_2, trust_competence_3
  - trust_integrity_1, trust_integrity_2, trust_integrity_3
  - trust_benevolence_1, trust_benevolence_2, trust_benevolence_3
  - trust_openness_1, trust_openness_2, trust_openness_3
  - trust_post, distrust_post, funding_perceptions
  - policy_role_mean, inst_trust_mean
  - belief_post, concern_mean, policy_general
  - policy_specific_mean, behavior_mean
  - donation_ams ($0-$10), newsletter_signup (0 or 1)

Usage:
  python generate_outcomes_qwen.py \\
    --profile_pool synthetic_profiles/profiles_pool.csv \\
    --model_path /projects/p32143/cache/qwen36_27b \\
    --batch_size 16 \\
    --output_dir raw_data_deposit \\
    --seed 2026

The script outputs:
  1. raw_output_<timestamp>.jsonl — LLM raw responses (for transparency)
  2. raw_data_<timestamp>.csv — Full Tier-1 submission-ready file
     (to be processed by scripts/clean.R)
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from questionnaire_parser import parse_intervention_texts


# ============================================================================
# CONSTANTS
# ============================================================================

# All 16 interventions (exact strings from submission_spec.R)
INTERVENTIONS = [
    "Corporate reliance",
    "Social justice",
    "Interview Prof. Maraun",
    "Funding",
    "Oil industry misinformation",
    "Measurement & modeling (1)",
    "Former skeptics",
    "High public trust",
    "Measurement & modeling (2)",
    "Peer-review",
    "Scientist community helpers",
    "Consensus",
    "Portrait Prof. Cherry",
    "Model accuracy",
    "Interview Prof. Sebille",
    "Extreme weather predictions",
]

# All outcome variables in submission order (submission_spec.R)
OUTCOME_VARS = [
    # Primary outcome
    "trust_multidimensional",
    # Trust sub-items (12)
    "trust_competence_1",
    "trust_competence_2",
    "trust_competence_3",
    "trust_integrity_1",
    "trust_integrity_2",
    "trust_integrity_3",
    "trust_benevolence_1",
    "trust_benevolence_2",
    "trust_benevolence_3",
    "trust_openness_1",
    "trust_openness_2",
    "trust_openness_3",
    # Other outcomes (12 more = 25 total)
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

# Value ranges for validation
SCALE_RANGES = {
    # 0-100 scales
    **{var: (0, 100) for var in [
        "trust_multidimensional", "trust_competence_1", "trust_competence_2", "trust_competence_3",
        "trust_integrity_1", "trust_integrity_2", "trust_integrity_3",
        "trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3",
        "trust_openness_1", "trust_openness_2", "trust_openness_3",
        "trust_post", "distrust_post", "funding_perceptions",
        "policy_role_mean", "inst_trust_mean", "belief_post", "concern_mean",
        "policy_general", "policy_specific_mean", "behavior_mean",
    ]},
    # 0-10 scale
    "donation_ams": (0, 10),
    # 0-1 scale (binary)
    "newsletter_signup": (0, 1),
}


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging to console and optionally to file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def format_with_chat_template(llm: "LLM", content: list, enable_thinking: bool = False) -> str:
    """Format prompt using the Qwen chat template."""
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )


def construct_survey_prompt(
    profile: Dict, intervention_text: str, logger: logging.Logger
) -> str:
    """
    Construct a survey prompt for a single profile.

    Args:
        profile: Dict with keys profile_id, condition, gender, age_band, race,
                education, income, party
        intervention_text: The full intervention/control stimulus text
        logger: Logger instance

    Returns:
        Formatted prompt text for the LLM
    """
    gender = profile["gender"]
    age_band = profile["age_band"]
    race = profile["race"]
    education = profile["education"]
    income = profile["income"]
    party = profile["party"]

    prompt = f"""You are a survey respondent with the following characteristics:
- Gender: {gender}
- Age: {age_band} years old
- Race/Ethnicity: {race}
- Education: {education}
- Household Income: {income}
- Political Party: {party}

You are now reading the following message about climate scientists:

{intervention_text}

---

After reading this message, please respond to the following survey questions.
For each question, provide your response on the specified scale.

TRUST IN CLIMATE SCIENTISTS (multiple items):

1. How incompetent or competent are most climate scientists? (0=Very incompetent, 100=Very competent):
2. How unintelligent or intelligent are most climate scientists? (0=Very unintelligent, 100=Very intelligent):
3. How unqualified or qualified are most climate scientists? (0=Very unqualified, 100=Very qualified):
4. How dishonest or honest are most climate scientists? (0=Very dishonest, 100=Very honest):
5. How unethical or ethical are most climate scientists? (0=Very unethical, 100=Very ethical):
6. How insincere or sincere are most climate scientists? (0=Very insincere, 100=Very sincere):
7. How unconcerned or concerned are most climate scientists about people's wellbeing? (0=Very unconcerned, 100=Very concerned):
8. How uneager or eager are most climate scientists to improve others' lives? (0=Very uneager, 100=Very eager):
9. How inconsiderate or considerate are most climate scientists of others' interests? (0=Very inconsiderate, 100=Very considerate):
10. How open, if at all, are most climate scientists to feedback? (0=Not open at all, 100=Very open):
11. How unwilling or willing are most climate scientists to be transparent? (0=Very unwilling, 100=Very willing):
12. How much or how little attention do climate scientists pay to other people's views? (0=Very little attention, 100=A great deal of attention):

SINGLE-ITEM TRUST & DISTRUST:
13. How much do you trust climate scientists? (0=not at all, 100=very strongly):
14. How much do you distrust climate scientists? (0=not at all, 100=very strongly):

FUNDING PERCEPTIONS:
15. Do you think the federal government spends too much, too little, or about the right amount on climate change research? (0=far too little, 100=far too much):

CLIMATE BELIEF:
16. How accurate do you think this statement is? "Human activities are causing climate change." (0=not at all accurate, 100=extremely accurate):

POLICY ROLE OF CLIMATE SCIENTISTS (average your agreement with):
17. Climate scientists should work closely with policy makers to integrate scientific results into policy-making. (0=Strongly disagree, 100=Strongly agree):
18. Climate scientists should actively advocate for specific policies. (0=Strongly disagree, 100=Strongly agree):
19. Climate scientists should communicate their findings to policy makers. (0=Strongly disagree, 100=Strongly agree):
20. Climate scientists should be more involved in the policy-making process. (0=Strongly disagree, 100=Strongly agree):

INSTITUTIONAL TRUST (average your trust in):
21. EPA (Environmental Protection Agency): (0=not at all, 100=very strongly):
22. NASA: (0=not at all, 100=very strongly):
23. NOAA: (0=not at all, 100=very strongly):
24. Universities and colleges: (0=not at all, 100=very strongly):
25. Federal government: (0=not at all, 100=very strongly):

CLIMATE CONCERN (average your response):
26. How concerned are you about climate change? (0=Not at all, 100=Extremely):
27. How serious a problem is climate change? (0=Not at all, 100=Extremely):
28. Relative to other issues facing the U.S., how important is climate change? (0=The least important issue, 100=The most important issue):

GENERAL POLICY SUPPORT:
29. How much do you oppose or support: "The U.S. government should do more to reduce global warming"? (0=Strongly oppose, 100=Strongly support):

SPECIFIC CLIMATE POLICIES (average your support):
30. Raising taxes on fossil fuels (e.g., gas, oil, coal) (0=Strongly oppose, 100=Strongly support):
31. Expanding infrastructure for public transportation (0=Strongly oppose, 100=Strongly support):
32. Increasing the use of sustainable energy such as wind and solar energy (0=Strongly oppose, 100=Strongly support):
33. Protecting forested and land areas (0=Strongly oppose, 100=Strongly support):
34. Increasing taxes on carbon-intensive foods (e.g., beef and dairy products) (0=Strongly oppose, 100=Strongly support):
35. Investing more in green jobs and businesses (0=Strongly oppose, 100=Strongly support):
36. Introducing laws to keep waterways and oceans clean (0=Strongly oppose, 100=Strongly support):

CLIMATE-FRIENDLY BEHAVIORS (average likelihood):
37. Eat less meat (0=Not likely at all, 100=Extremely likely):
38. Walk, bicycle, carpool, or take public transportation more often instead of driving by yourself (0=Not likely at all, 100=Extremely likely):
39. Install a solar panel (0=Not likely at all, 100=Extremely likely):
40. Go on less personal (non-business) air travel (0=Not likely at all, 100=Extremely likely):
41. Talk to friends and family about the importance of climate change (0=Not likely at all, 100=Extremely likely):
42. Donate to an environmental NGO (0=Not likely at all, 100=Extremely likely):

DONATION:
43. Of a $10 bonus, how much would you like to donate to the American Meteorological Society (AMS)? ($0-$10):

NEWSLETTER:
44. Would you like to subscribe to the "Talking Climate" newsletter? (0=No, 1=Yes):

---

Please provide your answers in the following format. Use ONLY the numbers, no additional text.

RESPONSE_FORMAT:
trust_competence_1: [0-100]
trust_competence_2: [0-100]
trust_competence_3: [0-100]
trust_integrity_1: [0-100]
trust_integrity_2: [0-100]
trust_integrity_3: [0-100]
trust_benevolence_1: [0-100]
trust_benevolence_2: [0-100]
trust_benevolence_3: [0-100]
trust_openness_1: [0-100]
trust_openness_2: [0-100]
trust_openness_3: [0-100]
trust_post: [0-100]
distrust_post: [0-100]
funding_perceptions: [0-100]
policy_role_1: [0-100]
policy_role_2: [0-100]
policy_role_3: [0-100]
policy_role_4: [0-100]
inst_trust_epa: [0-100]
inst_trust_nasa: [0-100]
inst_trust_noaa: [0-100]
inst_trust_universities: [0-100]
inst_trust_federal_gov: [0-100]
belief_post: [0-100]
concern_1: [0-100]
concern_2: [0-100]
concern_3: [0-100]
policy_general: [0-100]
policy_specific_1: [0-100]
policy_specific_2: [0-100]
policy_specific_3: [0-100]
policy_specific_4: [0-100]
policy_specific_5: [0-100]
policy_specific_6: [0-100]
policy_specific_7: [0-100]
behavior_meat: [0-100]
behavior_transport: [0-100]
behavior_solar: [0-100]
behavior_fly: [0-100]
behavior_talk: [0-100]
behavior_donate: [0-100]
donation_ams: [0-10]
newsletter_signup: [0 or 1]
"""

    return prompt.strip()


def parse_llm_output(output_text: str, logger: logging.Logger) -> Optional[Dict[str, float]]:
    """
    Parse the LLM output to extract outcome values.

    Expected format:
        trust_competence_1: 75
        trust_competence_2: 80
        ...

    Args:
        output_text: Raw LLM output text
        logger: Logger instance

    Returns:
        Dict of outcome variable values, or None if parsing fails
    """
    results = {}

    # Extract all lines with "key: value" format
    pattern = r"(\w+):\s*([0-9.]+)"
    matches = re.findall(pattern, output_text)

    if not matches:
        logger.warning("No key-value pairs found in output")
        return None

    for key, value_str in matches:
        try:
            value = float(value_str)
            results[key] = value
        except ValueError:
            logger.warning(f"Could not parse value for {key}: {value_str}")
            continue

    return results if results else None


def compute_composite_outcomes(raw_outcomes: Dict[str, float]) -> Dict[str, float]:
    """
    Compute composite outcomes (means of sub-items) from raw item responses.

    Args:
        raw_outcomes: Dict with all raw item responses

    Returns:
        Dict with raw items plus computed composites
    """
    outcomes = dict(raw_outcomes)

    # Trust subscales
    try:
        outcomes["trust_competence"] = sum([
            outcomes.get(f"trust_competence_{i}", 50) for i in [1, 2, 3]
        ]) / 3
        outcomes["trust_integrity"] = sum([
            outcomes.get(f"trust_integrity_{i}", 50) for i in [1, 2, 3]
        ]) / 3
        outcomes["trust_benevolence"] = sum([
            outcomes.get(f"trust_benevolence_{i}", 50) for i in [1, 2, 3]
        ]) / 3
        outcomes["trust_openness"] = sum([
            outcomes.get(f"trust_openness_{i}", 50) for i in [1, 2, 3]
        ]) / 3

        # Primary outcome: mean of 4 subscales
        outcomes["trust_multidimensional"] = sum([
            outcomes["trust_competence"],
            outcomes["trust_integrity"],
            outcomes["trust_benevolence"],
            outcomes["trust_openness"],
        ]) / 4
    except Exception as e:
        logging.warning(f"Error computing trust composites: {e}")

    # Policy role mean
    try:
        outcomes["policy_role_mean"] = sum([
            outcomes.get(f"policy_role_{i}", 50) for i in [1, 2, 3, 4]
        ]) / 4
    except Exception:
        outcomes["policy_role_mean"] = 50

    # Institutional trust mean
    try:
        outcomes["inst_trust_mean"] = sum([
            outcomes.get(f"inst_trust_{k}", 50)
            for k in ["epa", "nasa", "noaa", "universities", "federal_gov"]
        ]) / 5
    except Exception:
        outcomes["inst_trust_mean"] = 50

    # Concern mean
    try:
        outcomes["concern_mean"] = sum([
            outcomes.get(f"concern_{i}", 50) for i in [1, 2, 3]
        ]) / 3
    except Exception:
        outcomes["concern_mean"] = 50

    # Policy specific mean
    try:
        outcomes["policy_specific_mean"] = sum([
            outcomes.get(f"policy_specific_{i}", 50) for i in [1, 2, 3, 4, 5, 6, 7]
        ]) / 7
    except Exception:
        outcomes["policy_specific_mean"] = 50

    # Behavior mean
    try:
        outcomes["behavior_mean"] = sum([
            outcomes.get(f"behavior_{k}", 50)
            for k in ["meat", "transport", "solar", "fly", "talk", "donate"]
        ]) / 6
    except Exception:
        outcomes["behavior_mean"] = 50

    return outcomes


def validate_outcomes(outcomes: Dict[str, float], logger: logging.Logger) -> bool:
    """
    Validate that all outcome values are in the correct range.

    Args:
        outcomes: Dict of outcome values
        logger: Logger instance

    Returns:
        True if valid, False otherwise
    """
    all_valid = True
    for var, (min_val, max_val) in SCALE_RANGES.items():
        if var not in outcomes:
            logger.warning(f"Missing outcome variable: {var}")
            all_valid = False
            continue

        value = outcomes[var]
        if not (min_val <= value <= max_val):
            logger.warning(
                f"{var} = {value} out of range [{min_val}, {max_val}]"
            )
            all_valid = False

    return all_valid


def batched(items: List, batch_size: int):
    """Yield successive batches from items."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_intervention_texts(questionnaire_path: str) -> Dict[str, str]:
    """
    Load intervention stimulus texts from the questionnaire file.

    Args:
        questionnaire_path: Path to survey/questionnaire.txt

    Returns:
        Dict mapping condition names to full stimulus texts
    """
    interventions = parse_intervention_texts(questionnaire_path)
    if len(interventions) < 17:
        logging.warning(
            f"Expected 17 conditions from questionnaire_parser, got {len(interventions)}; "
            "profiles for missing conditions will fall back to the placeholder stimulus text."
        )
    return interventions


def main():
    parser = argparse.ArgumentParser(
        description="Generate outcomes for synthetic respondent profiles using Qwen LLM"
    )
    parser.add_argument(
        "--profile_pool",
        type=str,
        default="/projects/p32143/silicon-sample-submission/synthetic_profiles/profiles_pool.csv",
        help="Path to the synthetic profile pool CSV",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/projects/p32143/cache/qwen36_27b",
        help="Path to the Qwen model (local or HuggingFace ID)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for vLLM generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens for LLM generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0=deterministic, 1=max randomness)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Tensor parallelism size for vLLM",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="raw_data_deposit",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--max_profiles",
        type=int,
        default=None,
        help="Limit processing to first N profiles (for testing)",
    )
    parser.add_argument(
        "--questionnaire_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "survey" / "questionnaire.txt"),
        help="Path to survey/questionnaire.txt (for intervention stimulus text parsing)",
    )

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"generate_outcomes_{timestamp}.log"
    logger = setup_logging(str(log_file))

    logger.info("=" * 80)
    logger.info("SILICON SAMPLE BENCHMARK: OUTCOME GENERATION")
    logger.info("=" * 80)
    logger.info(f"Profile pool: {args.profile_pool}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output directory: {output_dir}")

    # Load profile pool
    logger.info("Loading profile pool...")
    df_profiles = pd.read_csv(args.profile_pool)
    if args.max_profiles:
        df_profiles = df_profiles.head(args.max_profiles)
    logger.info(f"Loaded {len(df_profiles)} profiles")

    # Load intervention texts
    interventions_dict = load_intervention_texts(args.questionnaire_path)

    # Initialize vLLM (import here to avoid dependency issues when module is imported elsewhere)
    from vllm import LLM, SamplingParams
    logger.info("Initializing vLLM engine (this may take a few moments)...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
        generation_config="vllm",
    )
    logger.info("vLLM engine initialized")

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Process profiles in batches
    all_results = []
    raw_output_file = output_dir / f"raw_output_{timestamp}.jsonl"

    logger.info(f"Processing {len(df_profiles)} profiles in batches of {args.batch_size}...")

    with open(raw_output_file, "w") as f_raw:
        for batch_idx, batch_rows in enumerate(
            batched([row for _, row in df_profiles.iterrows()], args.batch_size)
        ):
            batch_num = batch_idx + 1
            total_batches = (len(df_profiles) + args.batch_size - 1) // args.batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_rows)} profiles)...")

            # Construct prompts for this batch
            batch_prompts = []
            batch_profiles = []

            for profile_row in batch_rows:
                profile_dict = profile_row.to_dict()
                condition = profile_dict["condition"]

                # Get intervention text (or use placeholder)
                intervention_text = interventions_dict.get(
                    condition,
                    f"Condition: {condition}\n(Full intervention text should be provided)"
                )

                # Build survey prompt
                survey_prompt = construct_survey_prompt(
                    profile_dict, intervention_text, logger
                )

                # Format with chat template
                formatted_prompt = format_with_chat_template(llm, [
                    {"type": "text", "text": survey_prompt}
                ])

                batch_prompts.append({"prompt": formatted_prompt})
                batch_profiles.append(profile_dict)

            # Generate outputs
            try:
                outputs = llm.generate(batch_prompts, sampling)

                for idx, (output, profile_dict) in enumerate(zip(outputs, batch_profiles)):
                    profile_id = profile_dict["profile_id"]
                    condition = profile_dict["condition"]

                    raw_text = output.outputs[0].text.strip() if output.outputs else ""

                    # Parse outcomes
                    raw_outcomes = parse_llm_output(raw_text, logger)
                    if raw_outcomes is None:
                        logger.warning(f"{profile_id}: Failed to parse LLM output")
                        raw_outcomes = {}

                    # Compute composites
                    all_outcomes = compute_composite_outcomes(raw_outcomes)

                    # Validate
                    valid = validate_outcomes(all_outcomes, logger)
                    if not valid:
                        logger.warning(f"{profile_id}: Some outcomes out of range; using clipped values")
                        for var, (min_val, max_val) in SCALE_RANGES.items():
                            if var in all_outcomes:
                                all_outcomes[var] = max(min_val, min(max_val, all_outcomes[var]))

                    # Save raw output
                    raw_record = {
                        "profile_id": profile_id,
                        "condition": condition,
                        "raw_llm_output": raw_text[:500],  # Save first 500 chars
                        "parsed_outcomes": all_outcomes,
                    }
                    f_raw.write(json.dumps(raw_record) + "\n")

                    # Prepare for final CSV
                    row_data = dict(profile_dict)
                    row_data.update({
                        var: all_outcomes.get(var, 50)  # Default to midpoint if missing
                        for var in OUTCOME_VARS
                    })
                    all_results.append(row_data)

                    if (idx + 1) % 5 == 0:
                        logger.debug(f"  Processed {idx + 1}/{len(batch_rows)} in batch {batch_num}")

            except Exception as e:
                logger.error(f"Error during batch {batch_num} generation: {e}", exc_info=True)
                continue

    logger.info(f"Raw output saved to {raw_output_file}")

    # Assemble final Tier-1 CSV
    logger.info("Assembling final Tier-1 submission CSV...")

    # Column order: profile_id, condition, 6 demographics, 12 trust items, 13 outcomes
    column_order = [
        "profile_id", "condition",
        "gender", "age_band", "race", "education", "income", "party",
        # 12 trust sub-items
        "trust_competence_1", "trust_competence_2", "trust_competence_3",
        "trust_integrity_1", "trust_integrity_2", "trust_integrity_3",
        "trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3",
        "trust_openness_1", "trust_openness_2", "trust_openness_3",
        # 13 outcomes
        "trust_multidimensional", "trust_post", "distrust_post", "funding_perceptions",
        "policy_role_mean", "inst_trust_mean", "belief_post", "concern_mean",
        "policy_general", "policy_specific_mean", "behavior_mean",
        "donation_ams", "newsletter_signup",
    ]

    df_final = pd.DataFrame(all_results)
    df_final = df_final[column_order]

    output_csv = output_dir / f"tier1_submission_{timestamp}.csv"
    df_final.to_csv(output_csv, index=False)
    logger.info(f"Final Tier-1 CSV saved to {output_csv}")
    logger.info(f"  Rows: {len(df_final)}")
    logger.info(f"  Columns: {len(df_final.columns)}")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    for var in ["trust_multidimensional", "trust_post", "donation_ams", "newsletter_signup"]:
        if var in df_final.columns:
            col_data = df_final[var]
            logger.info(f"{var:30s}: mean={col_data.mean():.2f}, std={col_data.std():.2f}, "
                       f"min={col_data.min():.2f}, max={col_data.max():.2f}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ OUTCOME GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output files:")
    logger.info(f"  Raw output: {raw_output_file}")
    logger.info(f"  Tier-1 CSV: {output_csv}")
    logger.info(f"  Log file:   {log_file}")
    logger.info("\nNext steps:")
    logger.info(f"  1. Review raw outputs in {raw_output_file}")
    logger.info(f"  2. Check CSV columns and data ranges in {output_csv}")
    logger.info(f"  3. Run: scripts/clean.R {output_csv}")
    logger.info(f"  4. Run: make check")


if __name__ == "__main__":
    main()
