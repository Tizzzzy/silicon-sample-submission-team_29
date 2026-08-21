#!/usr/bin/env python3
"""
Parse intervention stimulus texts from survey/questionnaire.txt.

This module extracts the full stimulus text for all 17 conditions (control + 16 interventions)
from the raw questionnaire file, handling special cases (Extreme weather predictions with
state-adaptive variants, Consensus quiz) and stripping noise (sources, page breaks, etc.).

Used by both:
  - generate_outcomes_qwen.py (Tier-1 LLM outcome generation)
  - tier_2/generate_tier2_direct.py (Tier-2 direct group-level generation)

Public API:
  parse_intervention_texts(questionnaire_path, control_filler_index=1, extreme_weather_case=4)
    -> Dict[str, str]
    Returns {condition_name: cleaned_stimulus_text} for all 17 conditions.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def _read_lines(path: Union[str, Path]) -> List[str]:
    """Read a file and return list of lines (preserving original line endings for splitting)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _extract_condition_region(lines: List[str]) -> List[str]:
    """
    Extract the CONDITION region from the questionnaire file.
    Bounded by: '======' line with 'CONDITION' banner, and '------' fence before 'TRANSITION'.
    Returns the lines between these fences (content lines only).
    """
    condition_start_idx = None
    condition_end_idx = None

    # Look for the opening fence: a line of '=' chars that is part of the CONDITION banner block
    for i in range(len(lines) - 5):
        if (lines[i].startswith("=") and "CONDITION" in lines[i + 1] and
            i + 2 < len(lines) and lines[i + 2].startswith("=")):
            condition_start_idx = i + 3  # Content starts after the closing '=' fence
            break

    # Look for the closing fence: a line of '-' chars before TRANSITION
    # Start searching from AFTER the opening fence to avoid earlier TRANSITION sections
    for i in range(condition_start_idx, len(lines)):
        if lines[i].startswith("-" * 10) and i + 1 < len(lines) and "TRANSITION" in lines[i + 1]:
            condition_end_idx = i
            break

    if condition_start_idx is None:
        raise ValueError("Could not locate CONDITION region opening fence in questionnaire.txt")

    if condition_end_idx is None or condition_end_idx <= condition_start_idx:
        raise ValueError("Could not locate CONDITION region closing fence in questionnaire.txt")

    return lines[condition_start_idx:condition_end_idx]


def _split_into_blocks(region_lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Split the condition region into 19 blocks, each starting with '### Title'.
    Returns list of (title, body_lines) tuples.
    """
    blocks = []
    current_title = None
    current_body = []

    header_re = re.compile(r"^### (.+)$")

    for line in region_lines:
        match = header_re.match(line.strip())
        if match:
            # Save the previous block if it exists
            if current_title is not None:
                blocks.append((current_title, current_body))
            # Start a new block
            current_title = match.group(1)
            current_body = []
        else:
            if current_title is not None:
                current_body.append(line)

    # Don't forget the last block
    if current_title is not None:
        blocks.append((current_title, current_body))

    return blocks


def _clean_block_text(body_lines: List[str]) -> str:
    """
    Generic cleaner for most intervention blocks.
    Removes: page breaks, Source:/Sources: runs, [not displayed to participants] sections,
    and normalizes blank lines.
    """
    # Step 1: Remove page break markers
    page_break_re = re.compile(r"^\s*[-–—]+\s*page\s*break\s*[-–—]*\s*$", re.IGNORECASE)
    body_lines = [line for line in body_lines if not page_break_re.match(line)]

    # Step 2: Remove Source:/Sources: runs (that line + subsequent lines until blank line)
    cleaned = []
    skip_until_blank = False
    for line in body_lines:
        if re.match(r"^\s*Sources?:\s*", line):
            skip_until_blank = True
            continue
        if skip_until_blank:
            if line.strip() == "":
                skip_until_blank = False
            continue
        cleaned.append(line)
    body_lines = cleaned

    # Step 3: Truncate at [not displayed to participants]
    final = []
    for line in body_lines:
        if "[not displayed to participants]" in line:
            break
        final.append(line)
    body_lines = final

    # Step 4: Normalize blank lines and strip leading/trailing
    text = "\n".join(body_lines)
    # Collapse 3+ consecutive blank lines to 1
    text = re.sub(r"\n\n\n+", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def _parse_extreme_weather_block(body_lines: List[str], case: int) -> str:
    """
    Parse the "Extreme weather predictions" block, which is state-adaptive.

    Structure:
    - I. STIMULUS CASE ASSIGNMENT LOGIC (state -> case mapping)
    - II. STIMULUS (with Intervention Page 1, Page 2, Page 3)
      - Page 3 contains 4 case bodies, each preceded by a bare "Case N" header line

    For group-level simulation without a state field, we return the intro paragraph
    from the no-state branch plus the specified case body.
    """
    text = "\n".join(body_lines)

    # Find the "Prefer not to say" branch (the "IF" branch for no state)
    # This is in "Intervention Page 2" and should contain the generic intro
    if_match = re.search(
        r'IF state="Prefer not to say":\s*\n(.*?)(?:\n\s*ELSE:|$)',
        text,
        re.DOTALL
    )
    intro_paragraph = ""
    if if_match:
        intro_text = if_match.group(1).strip()
        # Take just the first paragraph (before the next blank line)
        intro_paragraph = intro_text.split("\n\n")[0]

    # Find the case body in "Intervention Page 3"
    # Case headers are bare lines like "Case 1", "Case 2", etc. (not the description lines from section I)
    case_header_re = re.compile(rf"^Case {case}\s*$", re.MULTILINE)
    case_match = case_header_re.search(text)
    if not case_match:
        logging.warning(f"Could not find 'Case {case}' header in Extreme weather predictions block; "
                       f"returning control fallback")
        return intro_paragraph

    case_start = case_match.end() + 1  # After the header line
    # Find the end: next "Case N" or "References [not displayed to participants]"
    next_case_re = re.compile(rf"^(?:Case [1-4]|References \[not displayed to participants\])", re.MULTILINE)
    next_match = next_case_re.search(text, case_start)

    if next_match:
        case_body = text[case_start : next_match.start()].strip()
    else:
        case_body = text[case_start:].strip()

    return (intro_paragraph + "\n\n" + case_body).strip()


def _parse_consensus_block(body_lines: List[str]) -> str:
    """
    Parse the "Consensus" block, which is an interactive quiz.

    This is NOT a plain passage — it has numbered estimate items, correct-answer reveals,
    per-item feedback with citations, then a closing "Summary:" paragraph.

    For group-level simulation, we use: intro paragraph + closing Summary paragraph,
    dropping the quiz mechanics, answers, and per-item feedback.
    """
    text = "\n".join(body_lines)

    # Intro = everything before "[Randomize...]"
    randomize_idx = text.find("[Randomize")
    if randomize_idx > 0:
        intro_text = text[:randomize_idx].strip()
        # Take the last paragraph of the intro (before the randomize line)
        intro_paragraphs = intro_text.split("\n\n")
        intro_paragraph = intro_paragraphs[-1]
    else:
        intro_paragraph = ""

    # Summary = everything after the literal line "Summary:" (which is the closing persuasive text)
    summary_match = re.search(r"^Summary:\s*$", text, re.MULTILINE)
    if summary_match:
        summary_start = summary_match.end() + 1
        summary_text = text[summary_start:].strip()
        # Take everything until the next "Source:" or [not displayed]
        summary_end_match = re.search(r"^(?:Source:|References)", summary_text, re.MULTILINE)
        if summary_end_match:
            summary_paragraph = summary_text[:summary_end_match.start()].strip()
        else:
            summary_paragraph = summary_text
    else:
        summary_paragraph = ""

    return (intro_paragraph + "\n\n" + summary_paragraph).strip()


def parse_intervention_texts(
    questionnaire_path: Union[str, Path],
    control_filler_index: int = 1,
    extreme_weather_case: int = 4,
) -> Dict[str, str]:
    """
    Parse intervention stimulus texts from the questionnaire file.

    Args:
        questionnaire_path: Path to survey/questionnaire.txt
        control_filler_index: Which of the 3 control filler texts to use (1, 2, or 3)
        extreme_weather_case: Which case variant of Extreme weather predictions (1, 2, 3, or 4)

    Returns:
        Dict mapping condition names (exactly matching submission_spec.R) to cleaned stimulus text.
        Keys: "control", "Corporate reliance", "Social justice", ... (16 interventions)
        All 17 keys guaranteed to be present.
    """
    lines = _read_lines(questionnaire_path)
    region = _extract_condition_region(lines)
    blocks = _split_into_blocks(region)

    if len(blocks) != 19:
        raise ValueError(
            f"Expected 19 '### ' blocks (3 control fillers + 16 interventions), "
            f"found {len(blocks)}"
        )

    result = {}
    control_bodies = []

    for title, body in blocks:
        if title.startswith("control — filler text"):
            control_bodies.append(body)
            continue

        # Special cases
        if title == "Extreme weather predictions":
            result[title] = _parse_extreme_weather_block(body, extreme_weather_case)
        elif title == "Consensus":
            result[title] = _parse_consensus_block(body)
        else:
            # Generic cleaning for other 14 interventions
            result[title] = _clean_block_text(body)

    # Handle control: pick one of the 3 filler texts
    if len(control_bodies) != 3:
        raise ValueError(f"Expected 3 control filler texts, found {len(control_bodies)}")

    if not (1 <= control_filler_index <= 3):
        raise ValueError(f"control_filler_index must be 1, 2, or 3; got {control_filler_index}")

    result["control"] = _clean_block_text(control_bodies[control_filler_index - 1])

    # Verify we have exactly 17 conditions
    if len(result) != 17:
        raise ValueError(f"Expected 17 conditions, got {len(result)}: {sorted(result.keys())}")

    return result


def main():
    """Self-test: load all conditions, verify structure, and report."""
    parser = argparse.ArgumentParser(
        description="Parse intervention texts from questionnaire.txt and validate structure."
    )
    parser.add_argument(
        "--questionnaire_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "survey" / "questionnaire.txt"),
        help="Path to survey/questionnaire.txt",
    )
    parser.add_argument(
        "--control_filler_index",
        type=int,
        default=1,
        help="Which control filler text to use (1, 2, or 3)",
    )
    parser.add_argument(
        "--extreme_weather_case",
        type=int,
        default=4,
        help="Which case variant for Extreme weather predictions (1, 2, 3, or 4)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Expected condition names (from submission_spec.R)
    EXPECTED_CONDITIONS = {
        "control",
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
    }

    NOISE_MARKERS = ("Source:", "Sources:", "page break", "[not displayed to participants]", "### ")

    try:
        texts = parse_intervention_texts(
            args.questionnaire_path,
            args.control_filler_index,
            args.extreme_weather_case,
        )

        # Validation
        all_ok = True

        if set(texts) != EXPECTED_CONDITIONS:
            missing = EXPECTED_CONDITIONS - set(texts)
            extra = set(texts) - EXPECTED_CONDITIONS
            print(f"FAIL: Condition set mismatch.")
            if missing:
                print(f"  Missing: {missing}")
            if extra:
                print(f"  Extra: {extra}")
            all_ok = False

        for cond in sorted(texts.keys()):
            text = texts[cond]
            bad_markers = [m for m in NOISE_MARKERS if m in text]
            is_empty = not text.strip()

            status = "OK " if (not bad_markers and not is_empty) else "BAD"
            print(
                f"{status} {cond:32s} "
                f"len={len(text):5d}  preview={text[:100]!r}"
            )

            if is_empty:
                print(f"    ERROR: Empty stimulus text")
                all_ok = False
            if bad_markers:
                print(f"    ERROR: Leaked noise markers {bad_markers}")
                all_ok = False

        print("\n" + ("=" * 80))
        if all_ok:
            print("✓ ALL CHECKS PASSED")
            return 0
        else:
            print("✗ SOME CHECKS FAILED")
            return 1

    except Exception as e:
        logging.error(f"Error during parsing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
