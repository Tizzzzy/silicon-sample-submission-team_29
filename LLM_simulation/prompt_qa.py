#!/usr/bin/env python3
"""
Prompt construction for the Silicon Sample Benchmark.

TWO SEPARATE THINGS, FROM TWO SEPARATE SOURCES
----------------------------------------------
1. HOW WE DESCRIBE THE RESPONDENT -- from Jahanparast, Hong & Chang, "What Do
   Large Language Models Know About Opinions?" (ICLR 2026). They condition on a
   respondent by writing the demographic as an ALREADY-ANSWERED survey question,
   so the conditioning sits in the same distribution as the target item:

       What is the highest level of schooling ... ?
       A. Less than high school
       ...
       E. College graduate/some postgrad
       Answer: E

   Their two alternates, BIO ("Description: The highest level of education I
   have completed is X") and PORTRAY ("Answer the following question as if
   ..."), are available via style=. This is the ONLY thing taken from the paper.

2. HOW WE ASK THE QUESTION -- from this challenge's own survey. Slider items are
   integers 0-100 shown with their endpoint labels and the survey's slider help
   text; the donation is a whole-dollar $0-$10 choice; the newsletter is Yes/No.
   A synthetic answer is then the same kind of object as a human answer, and
   needs no translation before scoring.

   An earlier draft binned the sliders into 11 lettered options and read the
   answer from next-token probabilities. That was borrowed from the paper, whose
   target questions came from Pew and natively had 2-3 choices. This survey asks
   for 0-100 integers, so that binning has been dropped.

HOW THIS DIFFERS FROM generate_outcomes_qwen.py
-----------------------------------------------
  * one item per call, not all 44 items in one response
  * decoding constrained by regex, so an out-of-range answer is impossible
  * all six demographics conditioned in the survey's own wording
  * the real stimulus text (stimuli.py), not a condition label

SETTINGS -- see PROMPTING.md and CHANGELOG.md:
  * demographics appear in survey order, all six at once (the paper conditions
    on ONE attribute at a time)
  * the stimulus is placed after the demographics and before the target item,
    wrapped in the survey's own TRANSITION wording
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from items import Item

# --- Response formats --------------------------------------------------------
# The survey's own formats, reproduced. Slider items are integers 0-100, the
# donation is a whole-dollar choice $0-$10, the newsletter is Yes/No. We elicit
# exactly these, so a synthetic answer is the same object as a human answer.
#
# The earlier draft binned sliders into 11 lettered options. That was imported
# from the ICLR paper, whose target questions came from Pew and natively had 2-3
# choices. It is not what this survey asks, and it is dropped. Only the paper's
# DEMOGRAPHIC conditioning format is kept (see demographic_preamble).
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The slider help text every slider item is shown with (questionnaire.txt).
SLIDER_HELP = ("Below is a range from 0 to 100. Click on any space within this "
               "range and a bar will appear. Feel free to move that bar around "
               "to the number that best represents your answer.")

# Regexes for constrained decoding, so the model cannot answer out of range.
REGEX_SLIDER = r"(100|[1-9]?[0-9])"
REGEX_DOLLARS = r"(10|[0-9])"
REGEX_BINARY = r"(Yes|No)"


def response_block(item: Item) -> Tuple[str, str, str]:
    """
    (scale line, answer cue, decoding regex) for one item's response format.
    """
    if item.scale == "binary":
        return ("", f"Answer ({item.high} or {item.low}):", REGEX_BINARY)
    if item.scale == "dollars":
        return ("Choices are whole dollars from $0 to $10.",
                "Answer (a whole number of dollars from 0 to 10):", REGEX_DOLLARS)
    anchors = f"0 = {item.low}"
    if item.mid:
        anchors += f", 50 = {item.mid}"
    anchors += f", 100 = {item.high}"
    return (f"{anchors}\n\n{SLIDER_HELP}",
            "Answer (a whole number from 0 to 100):", REGEX_SLIDER)


def parse_answer(text: str, item: Item) -> Optional[float]:
    """
    Turn the model's answer into a number on the item's own scale.
    Returns None if the answer is unusable -- never a midpoint fill.
    """
    text = text.strip()
    if item.scale == "binary":
        low = text.lower()
        if low.startswith("yes"):
            return 1.0
        if low.startswith("no"):
            return 0.0
        return None
    match = re.match(r"-?\d+", text)
    if not match:
        return None
    value = float(match.group())
    hi = 10.0 if item.scale == "dollars" else 100.0
    return value if 0.0 <= value <= hi else None


# --- Demographic conditioning ------------------------------------------------
# Survey wording and answer options, verbatim from questionnaire.txt, in the
# order a respondent met them. age_band is a derived moderator, so it is asked
# through the year-of-birth item it was derived from.
DEMOGRAPHIC_ITEMS: Sequence[Tuple[str, str, Sequence[str]]] = (
    ("gender", "What is your gender?",
     ("Male", "Female", "Other")),
    ("age_band", "What is your age?",
     ("18-29", "30-44", "45-59", "60+")),
    ("race", "Please select which race / ethnicity you most identify as.",
     ("White / Caucasian", "Black / African American", "Hispanic / Latino",
      "Asian / Asian American", "Other")),
    ("education", "What is the highest level of school that you have completed?",
     ("Less than high school", "High school diploma / GED",
      "Some college or Associate's degree", "Bachelor's degree",
      "Master's degree / Professional degree", "Doctorate degree / Ph.D.")),
    ("income", "What is your total yearly family/household income before taxes?",
     ("Less than $30,000", "$30,000 to $55,999", "$56,000 to $99,999",
      "$100,000 to $167,999", "$168,000 or more")),
    ("party", "Generally speaking, do you usually think of yourself as a "
              "Republican, a Democrat, an Independent, or what?",
     ("Republican", "Democrat", "Independent", "Other")),
)

_TRANSITION = ("You are now moving on to a different section of the study. Please "
               "pay close attention to the information you will be provided. "
               "Thank you.")
_FINAL_TRANSITION = ("You are now moving on to the final section of the study. "
                     "Please answer the following questions to the best of your "
                     "ability. Thank you.")


def _qa_block(question: str, options: Sequence[str], answer: str) -> str:
    """One answered question in the paper's QA format."""
    lines = [question]
    chosen = None
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
        if opt == answer:
            chosen = LETTERS[i]
    if chosen is None:
        raise ValueError(f"answer {answer!r} not among options for {question!r}")
    lines.append(f"Answer: {chosen}")
    return "\n".join(lines)


def demographic_preamble(profile: Dict[str, str], style: str = "qa") -> str:
    """Condition the model on the respondent, in one of the paper's 3 formats."""
    if style == "qa":
        return "\n\n".join(
            _qa_block(question, options, profile[key])
            for key, question, options in DEMOGRAPHIC_ITEMS
        )
    if style == "bio":
        facts = {
            "gender": "My gender is {}",
            "age_band": "I am {} years old",
            "race": "The race/ethnicity I most identify as is {}",
            "education": "The highest level of education I have completed is {}",
            "income": "My total yearly household income before taxes is {}",
            "party": "Politically I think of myself as {}",
        }
        body = ". ".join(facts[k].format(profile[k]) for k, _, _ in DEMOGRAPHIC_ITEMS)
        return ("Below you will be asked to provide a short description of "
                "yourself and then answer some questions.\n"
                f"Description: {body}.")
    if style == "portray":
        body = ", ".join(
            f"your {k.replace('_', ' ')} is {profile[k]}"
            for k, _, _ in DEMOGRAPHIC_ITEMS
        )
        return f"Answer the following question as if {body}."
    raise ValueError(f"unknown style {style!r}")


@dataclass(frozen=True)
class ItemPrompt:
    text: str
    regex: str        # constrains decoding to a valid answer for this item
    item_key: str
    max_tokens: int   # longest valid answer for this item, in tokens


def build_item_prompt(
    profile: Dict[str, str],
    stimulus: str,
    item: Item,
    style: str = "qa",
    include_stimulus: bool = True,
    extra_context: Optional[str] = None,
) -> ItemPrompt:
    """
    One prompt = one respondent + one stimulus + one scored item, ending in the
    item's answer cue. Decoding is constrained by ItemPrompt.regex.
    """
    parts = [demographic_preamble(profile, style)]

    if include_stimulus:
        parts.append(_TRANSITION)
        parts.append(stimulus)
        parts.append(_FINAL_TRANSITION)

    if extra_context:
        parts.append(extra_context)

    scale_line, cue, regex = response_block(item)
    q = []
    if item.intro:
        q.append(item.intro)
    q.append(item.text)
    if scale_line:
        q.append(scale_line)
    q.append(cue)
    parts.append("\n\n".join(q))

    return ItemPrompt(
        text="\n\n".join(parts),
        regex=regex,
        item_key=item.key,
        max_tokens=4 if item.scale != "binary" else 2,
    )


if __name__ == "__main__":
    import random
    import sys
    sys.path.insert(0, ".")
    from stimuli import parse_stimuli, stimulus_for, render_extreme_weather
    from items import BY_KEY

    stim = parse_stimuli("../survey/questionnaire.txt")
    rng = random.Random(2026)
    profile = {
        "profile_id": "p00042", "condition": "Peer-review",
        "gender": "Female", "age_band": "45-59",
        "race": "White / Caucasian", "education": "Bachelor's degree",
        "income": "$56,000 to $99,999", "party": "Republican",
    }
    p = build_item_prompt(
        profile, stimulus_for("Peer-review", stim, rng), BY_KEY["trust_post_1"]
    )
    print(p.text)
    print("\n--- decoding regex:", p.regex, "| max_tokens:", p.max_tokens)
