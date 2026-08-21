#!/usr/bin/env python3
"""
Offline checks for the outcome-generation pipeline. No GPU, no model needed.

Run from LLM_simulation/:   python test_pipeline.py

Covers the failure modes that got through last time: placeholder stimulus text,
the state-adaptive arm's scaffolding leaking into prompts, the reverse-coded
funding item, and missing answers being silently filled with the scale midpoint.
"""

import random
import sys

import pandas as pd

sys.path.insert(0, ".")
from generate_outcomes import SUBMISSION_COLUMNS, build_prompts_for_profile, composites
from items import ITEMS, load_newsletter_offer
from items import BY_KEY
from prompt_qa import parse_answer, response_block
from stimuli import parse_stimuli

QUESTIONNAIRE = "../survey/questionnaire.txt"
POOL = "../synthetic_profiles/profiles_pool.csv"


def main() -> None:
    stim = parse_stimuli(QUESTIONNAIRE)
    offer = load_newsletter_offer(QUESTIONNAIRE)
    profiles = pd.read_csv(POOL)
    rng = random.Random(0)

    # Two profiles per condition: prompts must build for all 17 arms.
    sample = profiles.groupby("condition").head(2)
    n = 0
    for _, row in sample.iterrows():
        prompts = build_prompts_for_profile(row.to_dict(), stim, offer, "qa", rng)
        assert len(prompts) == len(ITEMS), f"{len(prompts)} prompts, expected {len(ITEMS)}"
        for p in prompts:
            assert p.text.rstrip().endswith(":"), p.item_key
            assert "Answer" in p.text.rsplit("\n", 1)[-1], p.item_key
            assert p.regex and p.max_tokens > 0, p.item_key
        n += len(prompts)
    print(f"ok  built {n} prompts across {len(sample)} profiles")

    # No placeholder text survives.
    built = "\n".join(
        p.text for p in build_prompts_for_profile(
            profiles.iloc[0].to_dict(), stim, offer, "qa", rng))
    for bad in ("Full intervention text", "should be provided"):
        assert bad not in built, bad
    print("ok  no placeholder stimulus text in prompts")

    # The state-adaptive arm must not leak its authoring scaffolding.
    ew = profiles[profiles.condition == "Extreme weather predictions"].iloc[0].to_dict()
    ew_text = build_prompts_for_profile(ew, stim, offer, "qa", rng)[0].text
    for bad in ("STIMULUS CASE ASSIGNMENT", "not displayed to participants",
                "Prefer not to say", "Intervention page"):
        assert bad not in ew_text, bad
    print(f"ok  extreme-weather arm renders cleanly (state={ew['state']})")

    # Composites, including the reverse-coded funding item.
    vals = {it.key: 80.0 for it in ITEMS}
    vals["funding_5"] = 30.0
    vals["trust_competent_1"] = 20.0
    out = composites(vals)
    assert out["funding_perceptions"] == 70.0, out["funding_perceptions"]
    competence = (20 + 80 + 80) / 3
    assert abs(out["trust_multidimensional"] - (competence + 240) / 4) < 1e-9
    print("ok  composites correct (funding reversed 30 -> 70)")

    # A missing item must NOT become a midpoint fill.
    partial = {k: v for k, v in vals.items() if k != "trust_honest_1"}
    out2 = composites(partial)
    assert "trust_integrity_1" not in out2
    assert "trust_multidimensional" not in out2
    print("ok  missing item propagates as missing, not 50")

    # Items are asked in the survey's own response format, not a binned one.
    slider = build_prompts_for_profile(
        profiles.iloc[0].to_dict(), stim, offer, "qa", rng)[0]
    assert "whole number from 0 to 100" in slider.text
    assert "A." not in slider.text.split("Thank you.")[-1], "sliders must not be lettered"
    print("ok  sliders asked as integers 0-100, as the survey does")

    # Answer parsing: valid answers accepted, junk rejected, no midpoint fill.
    trust = BY_KEY["trust_post_1"]
    assert parse_answer("73", trust) == 73.0
    assert parse_answer("  0 ", trust) == 0.0
    assert parse_answer("100", trust) == 100.0
    assert parse_answer("101", trust) is None
    assert parse_answer("", trust) is None
    assert parse_answer("I would say quite a lot", trust) is None
    assert parse_answer("7", BY_KEY["donation"]) == 7.0
    assert parse_answer("11", BY_KEY["donation"]) is None
    assert parse_answer("Yes", BY_KEY["newsletter"]) == 1.0
    assert parse_answer("No", BY_KEY["newsletter"]) == 0.0
    assert parse_answer("Maybe", BY_KEY["newsletter"]) is None
    print("ok  answers parse; out-of-range and junk return None, not 50")

    # Every item's decoding regex admits its whole scale and nothing beyond it.
    import re as _re
    for it in ITEMS:
        _, _, rx = response_block(it)
        pat = _re.compile(rf"^{rx}$")
        if it.scale == "binary":
            valid, invalid = ["Yes", "No"], ["Maybe", "yes please"]
        elif it.scale == "dollars":
            valid, invalid = [str(v) for v in range(11)], ["11", "-1", "3.5"]
        else:
            valid, invalid = [str(v) for v in range(101)], ["101", "-1", "50.5"]
        for v in valid:
            assert pat.match(v), f"{it.key} regex rejects valid {v}"
        for v in invalid:
            assert not pat.match(v), f"{it.key} regex accepts invalid {v}"
    print("ok  decoding regexes admit the full scale and nothing else")

    # Every submission column is producible.
    full = composites({it.key: 50.0 for it in ITEMS})
    missing = [c for c in SUBMISSION_COLUMNS[8:] if c not in full]
    assert not missing, missing
    print(f"ok  all {len(SUBMISSION_COLUMNS[8:])} outcome columns produced")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
