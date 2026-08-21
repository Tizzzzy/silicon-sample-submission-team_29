#!/usr/bin/env python3
"""
The 44 scored survey items, with wording and endpoint labels taken verbatim
from codebook.csv / survey/questionnaire.txt.

Each item carries the block intro the human respondent saw above it, because
several items are only interpretable with it ("How much do you support or
oppose: — Protecting forested and land areas").

`submission_var` is the column the item feeds in the Tier-1 file; items sharing
a submission_var are averaged into it (see composites.py). `reverse=True` marks
the one item whose submission variable is 100 - raw (funding_perceptions).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Item:
    key: str                      # qualtrics_label
    intro: Optional[str]          # block intro shown above the item, if any
    text: str                     # item wording
    low: str                      # 0-end label
    high: str                     # 100-end label
    mid: Optional[str] = None     # midpoint label where the survey shows one
    scale: str = "slider"         # slider | dollars | binary
    submission_var: str = ""      # Tier-1 column this item feeds
    reverse: bool = False         # submission value is 100 - raw


_TRUST_INTRO = ("Please answer the following questions on how you perceive "
                "climate scientists.")
_INST_INTRO = "How much do you trust the following institutions?"
_POLICY_ROLE_INTRO = ("To what extent do you agree or disagree with the "
                      "following statements?")
_CONCERN_INTRO = "Please indicate your views on the following questions."
_POLICY_SPEC_INTRO = "How much do you support or oppose the following policies?"
_BEHAVIOR_INTRO = ("How likely are you to engage in the following activities in "
                   "the next twelve months?")

ITEMS: List[Item] = [
    # --- PRIMARY: multidimensional trust (12 items -> 4 subscales) ---
    Item("trust_competent_1", _TRUST_INTRO, "How incompetent or competent are most climate scientists?", "Very incompetent", "Very competent", submission_var="trust_competence_1"),
    Item("trust_intelligent_1", _TRUST_INTRO, "How unintelligent or intelligent are most climate scientists?", "Very unintelligent", "Very intelligent", submission_var="trust_competence_2"),
    Item("trust_qualified_1", _TRUST_INTRO, "How unqualified or qualified are most climate scientists?", "Very unqualified", "Very qualified", submission_var="trust_competence_3"),
    Item("trust_honest_1", _TRUST_INTRO, "How dishonest or honest are most climate scientists?", "Very dishonest", "Very honest", submission_var="trust_integrity_1"),
    Item("trust_ethical_1", _TRUST_INTRO, "How unethical or ethical are most climate scientists?", "Very unethical", "Very ethical", submission_var="trust_integrity_2"),
    Item("trust_sincere_1", _TRUST_INTRO, "How insincere or sincere are most climate scientists?", "Very insincere", "Very sincere", submission_var="trust_integrity_3"),
    Item("trust_concerned_1", _TRUST_INTRO, "How unconcerned or concerned are most climate scientists about people’s wellbeing?", "Very unconcerned", "Very concerned", submission_var="trust_benevolence_1"),
    Item("trust_improve_1", _TRUST_INTRO, "How uneager or eager are most climate scientists to improve others’ lives?", "Very uneager", "Very eager", submission_var="trust_benevolence_2"),
    Item("trust_considerate_1", _TRUST_INTRO, "How inconsiderate or considerate are most climate scientists of others’ interests?", "Very inconsiderate", "Very considerate", submission_var="trust_benevolence_3"),
    Item("trust_feedback_1", _TRUST_INTRO, "How open, if at all, are most climate scientists to feedback?", "Not open at all", "Very open", submission_var="trust_openness_1"),
    Item("trust_transparent_1", _TRUST_INTRO, "How unwilling or willing are most climate scientists to be transparent?", "Very unwilling", "Very willing", submission_var="trust_openness_2"),
    Item("trust_attention_1", _TRUST_INTRO, "How much or how little attention do climate scientists pay to other people's views?", "Very little attention", "A great deal of attention", submission_var="trust_openness_3"),

    # --- SECONDARY ---
    Item("funding_5", None, "Do you think the federal government is spending too much, too little or about the right amount of money on climate change research?", "far too little", "far too much", mid="about the right amount", submission_var="funding_perceptions", reverse=True),
    Item("inst_trust_epa_1", _INST_INTRO, "Environmental Protection Agency (EPA)", "not at all", "very strongly", submission_var="inst_trust_mean"),
    Item("inst_trust_nasa_1", _INST_INTRO, "National Aeronautics and Space Administration (NASA)", "not at all", "very strongly", submission_var="inst_trust_mean"),
    Item("inst_trust_noaa_1", _INST_INTRO, "National Oceanic and Atmospheric Administration (NOAA)", "not at all", "very strongly", submission_var="inst_trust_mean"),
    Item("inst_trust_uni_1", _INST_INTRO, "Universities and colleges", "not at all", "very strongly", submission_var="inst_trust_mean"),
    Item("inst_trust_gov_1", _INST_INTRO, "Federal government", "not at all", "very strongly", submission_var="inst_trust_mean"),
    Item("policy_1_1", _POLICY_ROLE_INTRO, "Climate scientists should work closely with policy makers to integrate scientific results into policy-making.", "Strongly disagree", "Strongly agree", submission_var="policy_role_mean"),
    Item("policy_2_1", _POLICY_ROLE_INTRO, "Climate scientists should actively advocate for specific policies.", "Strongly disagree", "Strongly agree", submission_var="policy_role_mean"),
    Item("policy_3_1", _POLICY_ROLE_INTRO, "Climate scientists should communicate their findings to policy makers.", "Strongly disagree", "Strongly agree", submission_var="policy_role_mean"),
    Item("policy_4_1", _POLICY_ROLE_INTRO, "Climate scientists should be more involved in the policy-making process.", "Strongly disagree", "Strongly agree", submission_var="policy_role_mean"),
    Item("trust_post_1", None, "How much do you trust climate scientists?", "not at all", "very strongly", submission_var="trust_post"),
    Item("distrust_1", None, "How much do you distrust climate scientists?", "not at all", "very strongly", submission_var="distrust_post"),
    Item("donation", None, "Of the $10 bonus, how much would you like to donate to the American Meteorological Society (AMS)?", "$0", "$10", scale="dollars", submission_var="donation_ams"),
    Item("newsletter", None, "Did you subscribe to the “Talking Climate” newsletter on the previous page?", "No", "Yes", scale="binary", submission_var="newsletter_signup"),

    # --- TERTIARY ---
    Item("belief_post_1", None, "How accurate do you think this statement is? “Human activities are causing climate change.”", "not at all accurate", "extremely accurate", submission_var="belief_post"),
    Item("concern_1_1", _CONCERN_INTRO, "How concerned are you about climate change?", "Not at all", "Extremely", submission_var="concern_mean"),
    Item("concern_2_1", _CONCERN_INTRO, "How serious a problem is climate change?", "Not at all", "Extremely", submission_var="concern_mean"),
    Item("concern_3_1", _CONCERN_INTRO, "Relative to other issues facing the U.S., how important is climate change?", "The least important issue", "The most important issue", submission_var="concern_mean"),
    Item("policy_general_1", None, "How much do you oppose or support: “The U.S. government should do more to reduce global warming”", "Strongly oppose", "Strongly support", submission_var="policy_general"),
    Item("policy_specific_1_1", _POLICY_SPEC_INTRO, "Raising taxes on fossil fuels (e.g., gas, oil, coal)", "Strongly oppose", "Strongly support", submission_var="policy_specific_mean"),
    Item("policy_specific_2_1", _POLICY_SPEC_INTRO, "Expanding infrastructure for public transportation", "Strongly oppose", "Strongly support", submission_var="policy_specific_mean"),
    Item("policy_specific_3_1", _POLICY_SPEC_INTRO, "Increasing the use of sustainable energy such as wind and solar energy", "Strongly oppose", "Strongly support", submission_var="policy_specific_mean"),
    Item("policy_specific_4_1", _POLICY_SPEC_INTRO, "Protecting forested and land areas", "Strongly oppose", "Strongly support", submission_var="policy_specific_mean"),
    Item("policy_specific_5_1", _POLICY_SPEC_INTRO, "Increasing taxes on carbon-intensive foods (e.g., beef and dairy products)", "Strongly oppose", "Strongly support", submission_var="policy_specific_mean"),
    Item("policy_specific_6_1", _POLICY_SPEC_INTRO, "Investing more in green jobs and businesses", "Strongly oppose", "Strongly support", submission_var="policy_specific_mean"),
    Item("policy_specific_7_1", _POLICY_SPEC_INTRO, "Introducing laws to keep waterways and oceans clean", "Strongly oppose", "Strongly support", submission_var="policy_specific_mean"),
    Item("individual_meat_1", _BEHAVIOR_INTRO, "Eat less meat", "Not likely at all", "Extremely likely", submission_var="behavior_mean"),
    Item("individual_transport_1", _BEHAVIOR_INTRO, "Walk, bicycle, carpool, or take public transportation more often instead of driving by yourself", "Not likely at all", "Extremely likely", submission_var="behavior_mean"),
    Item("individual_solar_1", _BEHAVIOR_INTRO, "Install a solar panel", "Not likely at all", "Extremely likely", submission_var="behavior_mean"),
    Item("individual_fly_1", _BEHAVIOR_INTRO, "Go on less personal (non-business) air travel", "Not likely at all", "Extremely likely", submission_var="behavior_mean"),
    Item("individual_talk_1", _BEHAVIOR_INTRO, "Talk to friends and family about the importance of climate change", "Not likely at all", "Extremely likely", submission_var="behavior_mean"),
    Item("individual_donate_1", _BEHAVIOR_INTRO, "Donate to an environmental NGO", "Not likely at all", "Extremely likely", submission_var="behavior_mean"),
]

BY_KEY = {it.key: it for it in ITEMS}

# The newsletter item refers back to an offer page shown immediately before it.
# A one-item-per-call design must supply that page or the item is unanswerable.
NEWSLETTER_OFFER = None  # filled by load_newsletter_offer()


def load_newsletter_offer(questionnaire_path: str = "survey/questionnaire.txt") -> str:
    """Pull the verbatim newsletter offer page out of the questionnaire."""
    from pathlib import Path
    text = Path(questionnaire_path).read_text(encoding="utf-8")
    start = text.index("## Subscription to climate science newsletter")
    end = text.index("# TERTIARY OUTCOMES", start)
    return text[start:end]


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(ITEMS)} items")
    for var, n in sorted(Counter(i.submission_var for i in ITEMS).items()):
        print(f"  {n:2d}  {var}")
