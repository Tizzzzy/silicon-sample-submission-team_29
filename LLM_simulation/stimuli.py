#!/usr/bin/env python3
"""
Parse the condition stimulus texts out of survey/questionnaire.txt.

Replaces the hardcoded placeholder dict in generate_outcomes_qwen.py, which
supplied real text for only 3 of 17 conditions.

The questionnaire's CONDITION section holds 19 blocks introduced by a "### "
heading: three control filler texts and the 16 intervention stimuli. Control
respondents each saw exactly ONE of the three fillers, assigned at random.
"""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional

# Exact condition labels a submission must carry (submission_spec.R).
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

_CONTROL_HEADING = re.compile(r"^### control — filler text \d of 3: (.+)$")
_HEADING = re.compile(r"^### (.+)$")
_SECTION_START = "CONDITION  (each respondent sees exactly ONE"
_SECTION_END = "POST-TREATMENT"


def parse_stimuli(questionnaire_path: str) -> Dict[str, str]:
    """
    Return {condition_label: stimulus_text} for the 16 interventions, plus
    {"control:<slug>": text} for each of the three control fillers.

    Raises if any of the 16 interventions or 3 fillers is missing, so a
    truncated questionnaire fails loudly instead of silently degrading the run.
    """
    lines = Path(questionnaire_path).read_text(encoding="utf-8").splitlines()

    # Narrow to the CONDITION section so headings elsewhere cannot collide.
    start = next(i for i, ln in enumerate(lines) if _SECTION_START in ln)
    end = next(
        (i for i, ln in enumerate(lines[start:], start) if _SECTION_END in ln),
        len(lines),
    )

    blocks: Dict[str, List[str]] = {}
    current = None
    for line in lines[start:end]:
        heading = _HEADING.match(line)
        if heading:
            control = _CONTROL_HEADING.match(line)
            if control:
                slug = control.group(1).strip().lower().replace(" ", "_")
                current = f"control:{slug}"
            else:
                current = heading.group(1).strip()
            blocks[current] = []
            continue
        if current is not None:
            blocks[current].append(line)

    stimuli = {
        key: "\n".join(body).strip()
        for key, body in blocks.items()
        if "\n".join(body).strip()
    }

    missing = [c for c in INTERVENTIONS if c not in stimuli]
    if missing:
        raise ValueError(f"missing intervention text for: {missing}")
    fillers = [k for k in stimuli if k.startswith("control:")]
    if len(fillers) != 3:
        raise ValueError(f"expected 3 control fillers, found {len(fillers)}: {fillers}")

    return stimuli


def stimulus_for(condition: str, stimuli: Dict[str, str], rng: random.Random) -> str:
    """Text shown to one respondent; control draws one filler at random."""
    if condition == "control":
        fillers = sorted(k for k in stimuli if k.startswith("control:"))
        return stimuli[rng.choice(fillers)]
    return stimuli[condition]



# ---------------------------------------------------------------------------
# "Extreme weather predictions" is STATE-ADAPTIVE
# ---------------------------------------------------------------------------
# The questionnaire block for this arm is authoring scaffolding, not a stimulus:
# it holds the state->case mapping, the state question, two intro variants, and
# four case texts. A respondent sees one intro paragraph plus ONE case text.
# Feeding the raw block verbatim (as a naive parser would) shows the respondent
# the experiment's own design notes.

_CASE_LABELS = {
    1: "states with high or recurrent flood risk",
    2: "states with high or increasing wildfire risk",
    3: "states with severe cold, snow, ice, or blizzards",
}

_STATE_TO_CASE = {s: 1 for s in [
    "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Illinois", "Indiana",
    "Iowa", "Kansas", "Kentucky", "Louisiana", "Maryland", "Mississippi", "Missouri",
    "Nebraska", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Pennsylvania",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Virginia",
    "West Virginia", "Washington, D.C.",
]}
_STATE_TO_CASE.update({s: 2 for s in [
    "Alaska", "Arizona", "California", "Colorado", "Idaho", "Montana", "Nevada",
    "New Mexico", "Oregon", "Utah", "Washington", "Wyoming", "Hawaii",
]})
_STATE_TO_CASE.update({s: 3 for s in [
    "Connecticut", "Maine", "Massachusetts", "Michigan", "Minnesota",
    "New Hampshire", "New Jersey", "New York", "Rhode Island", "Vermont",
    "Wisconsin",
]})

_INTRO_GENERIC = (
    "You are living in the United States, a country facing risks by more and more "
    "extreme weather events. Please read the text on the following page carefully. "
    "It describes a real project in the U.S., working particularly on reducing the "
    "risks from these hazards by helping communities prepare for extreme weather."
)
_INTRO_STATE = (
    "You reported that you are currently living in {state}, one of several {case}. "
    "Please read the text on the following page carefully. It describes a real "
    "project in the U.S., working particularly on reducing the risks from these "
    "hazards by helping communities prepare for extreme weather."
)


def parse_extreme_weather_cases(block: str) -> Dict[int, str]:
    """Split the state-adaptive block into its four case texts."""
    page3 = block.split("Intervention page 3", 1)[1]
    page3 = page3.split("References [not displayed", 1)[0]
    parts = re.split(r"^Case ([1-4])\s*$", page3, flags=re.MULTILINE)
    cases = {}
    for i in range(1, len(parts) - 1, 2):
        cases[int(parts[i])] = parts[i + 1].strip()
    if set(cases) != {1, 2, 3, 4}:
        raise ValueError(f"expected cases 1-4, got {sorted(cases)}")
    return cases


def render_extreme_weather(block: str, state: Optional[str]) -> str:
    """
    Build what one respondent in the 'Extreme weather predictions' arm sees:
    the intro paragraph for their state, then that state's case text.

    state=None reproduces the "Prefer not to say" branch (generic intro, case 4).
    """
    cases = parse_extreme_weather_cases(block)
    case_num = _STATE_TO_CASE.get(state) if state else None
    if case_num is None:
        return f"{_INTRO_GENERIC}\n\n{cases[4]}"
    intro = _INTRO_STATE.format(state=state, case=_CASE_LABELS[case_num])
    return f"{intro}\n\n{cases[case_num]}"


ALL_STATES = sorted(_STATE_TO_CASE)


if __name__ == "__main__":
    import sys

    s = parse_stimuli(sys.argv[1] if len(sys.argv) > 1 else "survey/questionnaire.txt")
    for key in sorted(s):
        words = len(s[key].split())
        print(f"{words:5d} words  {key}")
    print(f"\n{len(s)} blocks parsed")
