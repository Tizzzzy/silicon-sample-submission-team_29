# Default prompt

Two independent choices, from two different sources.

## 1. How we describe the respondent — from the ICLR 2026 paper

Jahanparast, Hong & Chang, *What Do Large Language Models Know About Opinions?*,
ICLR 2026 (UC Berkeley). <https://github.com/schang-lab/llm-opinions>

Their contribution we use: write each demographic as an **already-answered
survey question**, so the conditioning sits in the same distribution as the
question being asked.

```
What is the highest level of school that you have completed?
A. Less than high school
...
D. Bachelor's degree
Answer: D
```

Two alternates from their Table 2 are also implemented, selectable with `style=`:
**BIO** ("Description: The highest level of education I have completed is …") and
**PORTRAY** ("Answer the following question as if …").

**This is the only thing taken from the paper.** We are not using its probing
method — that needs human answer distributions to train on, and this study is
blind.

## 2. How we ask the question — from this challenge's survey

Items are asked in the survey's own response format:

| item type | format | constrained to |
|---|---|---|
| 44 slider items | integer 0–100, with endpoint labels and the survey's slider help text | `(100\|[1-9]?[0-9])` |
| donation | whole dollars, $0–$10 | `(10\|[0-9])` |
| newsletter | Yes / No | `(Yes\|No)` |

Decoding is constrained by regex, so an out-of-range or malformed answer is
impossible rather than something to clean up afterwards. A synthetic answer is
then the same kind of object as a human answer and needs no translation.

> An earlier draft binned the sliders into 11 lettered options (A–K at 0, 10, …
> 100) and read the answer from next-token probabilities. That was borrowed from
> the paper, whose target questions came from Pew and natively had 2–3 choices.
> This survey asks for 0–100 integers. The binning was dropped.

Run `python prompt_qa.py` to print a complete example prompt.

## Settings — all configurable, none forced

1. **`--temperature 1.0` (OPEN).** With one item per call and a fixed prompt,
   temperature is the *only* thing that makes two respondents with the same six
   demographics in the same condition answer differently. At 0 they would be
   identical. At 1 the spread is whatever the model's own distribution gives,
   which is not guaranteed to match human spread. Needs a decision.
2. **All six demographics conditioned at once**, in survey order. The paper
   conditions on one attribute at a time (22 single-attribute groups). Tier 1
   needs a whole respondent, so we stack them. Whether the model handles six as
   well as one is an open question the paper does not answer.
3. **Age is asked directly**, not through year of birth. The survey asks
   `year_birth` and derives `age_band`; the profile pool stores only the band.
4. **Stimulus placed between demographics and item**, wrapped in the survey's own
   two TRANSITION paragraphs. The paper has no treatment at all, so this
   placement is ours.
5. **Stimulus re-sent on every one of the 44 calls.** Prefix caching absorbs most
   of the cost, since the prefix is identical within a respondent.
6. **Control respondents draw one of the three filler texts at random**, matching
   the survey. Seeded per respondent.
7. **`Extreme weather predictions` is rendered per respondent from an assigned
   state** (`render_extreme_weather`). The questionnaire block for this arm is
   authoring scaffolding — mapping tables, both intro variants, all four case
   texts — and must not be fed verbatim. States are drawn in proportion to
   resident population.
8. **Newsletter item is given its offer page.** The scored item asks whether the
   respondent subscribed *on the previous page*, so the offer text is passed via
   `extra_context=` (`items.load_newsletter_offer()`).

## Corrections to the previous pipeline folded in here

- **`funding_perceptions` is reverse-coded**: the submission variable is
  `100 − funding_5`. The old prompt asked the raw direction and stored it
  unreversed, flipping the sign of that outcome. `Item.reverse` marks it.
- Item wording now matches `codebook.csv` verbatim, including the block intros
  ("How much do you support or oppose the following policies?") that several
  items are meaningless without.
