# Change log

A running record of what we changed and why, so the work can be handed back with
the reasoning attached. Newest first. Entries marked **DECISION** are settings we
picked that nobody had specified; they are open to revisiting.

---

## 2026-08-20 (later still) — Dry-run mode, and two bugs it caught

The real smoke run needs the cluster: vLLM does not run on a Mac and the 27B
model is not local. But everything except the model call can be tested without a
GPU, so `generate_outcomes.py --dry_run` now runs the whole pipeline against a
stub model (`stub_llm.py`) that returns format-valid answers plus a controllable
share of unparseable ones.

It found two bugs that would both have surfaced on the cluster:

1. **`chat_wrap` was undefined.** Lost when the vLLM setup block was
   restructured. The run crashed on the first batch. This would have failed the
   real run immediately.
2. **Column order was wrong.** `trust_multidimensional` belongs at position 9,
   *before* the twelve trust sub-items — that is the order in
   `scripts/lib/submission_spec.R` and in the organizers' example file. Both the
   old pipeline and the rebuild had it at position 21. Fixed.

Verified at full production scale (36,000 profiles, 1,584,000 calls, 16 seconds
without a model):

- 36,000 rows out, profile ids matching the pool exactly, no duplicates, no NAs
- condition counts match the pool; 2,000 per intervention, 4,000 control
- composites recomputed by hand from the raw log agree with the CSV
- `funding_perceptions` reversal confirmed on real output (raw 87 -> 13)
- checked against `submission_spec.R` in R: exact column order match, all 17
  condition labels valid, all six moderators' levels valid, every outcome inside
  its native range

A dry run proves the plumbing, not the science. Answers are random. It can never
produce a submission.

---

## 2026-08-20 (later still) — Prior work on the temperature / variance question

Looked up whether the sampling-vs-averaging choice has been studied. It has, and
the findings bear directly on the open temperature decision.

**What the benchmark scores.** `FAQ.md` is explicit: "only point estimates are
scored." Tier-1 rows are reduced to cell means and effects. Within-cell variance
realism is therefore not scored directly.

**Temperature probably will not produce the variation we want.**
Jang, Lee & Kim (KAIST 2026), *Instruction-Tuned Language Models Cannot Sample
from Distributions They Can Describe*: repeated calls on identical
(persona, item) pairs returned identical answers 57% of the time. Temperature
and top-p adjustments were ineffective — logit gaps reached 14+ nats, beyond
what temperature scaling can recover. Asking the model to *describe* the
distribution instead cut error from 0.46 to 0.22 total-variation distance on 100
OpinionQA items.

**Mode collapse is well documented but fixing it does not fix means.**
Bisbee et al. (2024, *Political Analysis*): ChatGPT recovered feeling-thermometer
means reasonably well but showed far less variation than humans; 48% of
regression coefficients differed significantly from the human data and 32%
flipped sign. Heath & Alexander (2026) correct mode collapse with Semantic
Similarity Rating (KL 0.61–1.97 → 0.07–0.13), but report that **absolute error of
synthetic means was largely unchanged**.

**Implication for us.** Because only point estimates are scored, and the one
paper that successfully repaired variance did not thereby improve means,
chasing realistic within-cell spread is unlikely to be where our score comes
from. The real risk of collapse is different: if each demographic cell collapses
to its *modal* answer rather than the mean of the model's belief, the aggregate
cell mean is biased. That is worth measuring.

**Concrete next step.** The smoke run should count how often two respondents with
identical demographics in the same condition receive identical answers. If that
rate is high — the KAIST paper suggests it will be — `--temperature 1.0` is doing
nothing and the design needs revisiting, most likely toward eliciting a
distribution per demographic cell and drawing respondents from it.

Not decided. Flagged for discussion.

---

## 2026-08-20 (later) — Response format reverted to the survey's own

**Reversal of an earlier decision in this same session.** Sliders had been binned
into 11 lettered options (A–K at 0, 10, … 100), with the answer read from the
model's next-token probability over the letters. That is dropped.

Why it was wrong. The binning was borrowed from the ICLR paper, on the grounds
that the paper "only ever used 2 or 3 options." That reasoning conflated the two
halves of the paper's prompt. The 2-to-3-option limit applied to the paper's
**target** questions, which came from Pew's American Trends Panel and natively
have two or three choices. Their **demographic** conditioning blocks were not
limited that way — their education example runs A through F. So a constraint
from the half of the paper we are not copying was applied to the half we are.

More importantly, the paper does not get to set our response format. The
challenge does. `survey/questionnaire.txt` and `codebook.csv` both state that
slider items are **integers 0–100, no decimals**; the donation is whole dollars
$0–$10; the newsletter is Yes/No. Human respondents answer on a 101-point scale,
so synthetic respondents standing in for them should too. An 11-point answer
would have needed translating before scoring, and would have thrown away most of
the scale's resolution at the individual level.

What we do now: ask each item exactly as the survey asks it, including endpoint
labels and the survey's own slider help text, and constrain decoding with a
regex so an out-of-range or malformed answer cannot occur.

| item type | asked as | decoding constrained to |
|---|---|---|
| 44 sliders | integer 0–100 | `(100\|[1-9]?[0-9])` |
| donation | whole dollars 0–10 | `(10\|[0-9])` |
| newsletter | Yes / No | `(Yes\|No)` |

**Only the demographic conditioning format is still taken from the paper.**

Consequences:

- Answers are no longer read from a probability distribution, so the
  distribution-vs-sampling choice (old decision 4) no longer exists.
- **Temperature is now load-bearing and is an open decision.** It is the only
  thing that makes two respondents with identical demographics in the same
  condition answer differently. At temperature 0 they are identical, which would
  reproduce the mode collapse of the first run. Currently set to 1.0, which is
  not a justified value, just the neutral one.
- Output grows from 1 token per call to at most 4. Negligible: ~4M output tokens
  across the full run, against 22M for the old single-call design.
- `test_pipeline.py` now checks that every item's regex admits its whole scale
  (all 101 values for a slider) and rejects 101, −1, and 50.5.

---

## 2026-08-20 — Rebuild of the outcome-generation pipeline

### Why

The first run (`raw_data_deposit/*_20260817_011246.*`, 9,000 rows) did not
measure what it was meant to measure. Three separate problems, in order of
severity.

### 1. The stimulus texts were placeholders (run-invalidating)

`generate_outcomes_qwen.py` built its intervention texts from a hardcoded dict
holding real text for only 3 of the 17 conditions. The other 14 got:

```
Condition: High public trust
(Full intervention text should be provided)
```

The control condition got an invented sentence instead of the real filler text.
So for 14 arms the model saw a **condition name and nothing else**. The
condition means still separated (control 68.3, "High public trust" 78.8, "Oil
industry misinformation" 61.8), because a label leaks the gist — which is why
the run looked plausible.

The texts were never missing. `survey/questionnaire.txt` ships all 16
intervention texts plus the 3 control fillers; the README lists intervention
texts among what the benchmark provides. The function had a live branch that
logged `"Reading intervention texts from survey/questionnaire.txt"` and then did
nothing with the file, which is likely why nobody noticed.

**Fix:** `LLM_simulation/stimuli.py` parses the questionnaire. 19 blocks, all
verified present, and it raises if any is missing rather than degrading quietly.

### 2. `funding_perceptions` was sign-flipped

`codebook.csv` defines the submission variable as `100 − funding_5`. The old
prompt asked the raw direction (0 = far too little, 100 = far too much) and
wrote it out unreversed, so that outcome pointed backwards in the whole file.

**Fix:** `Item.reverse` in `items.py`, applied in `composites()`.

### 3. `Extreme weather predictions` cannot be pasted verbatim

That arm is state-adaptive. Its questionnaire block is 1,657 words of authoring
scaffolding: the state → hazard mapping table, the 51-item state dropdown, both
intro variants, all four case texts, and the reference list. A respondent sees
one intro plus one ~300-word case. Feeding the block raw would have shown a
synthetic respondent the experiment's own design notes.

**Fix:** `render_extreme_weather()` assembles the per-respondent version.

---

### Prompt strategy: one item per call, answers read from token probabilities

Following Jahanparast, Hong & Chang, *What Do Large Language Models Know About
Opinions?* (ICLR 2026). We use the paper's **prompting format**, not its probing
method — probes need human answer distributions to train on, and this study is
blind. Not pursuing probes.

What changed:

| | old | new |
|---|---|---|
| calls per respondent | 1 (all 44 items) | 44 (one item each) |
| answer extraction | model writes numbers, regex-parsed | next-token probability over option letters |
| response scale | free numeric | 11 lettered options (0, 10, … 100) |
| demographics | 6 bullet points in a persona header | 6 answered survey questions, survey wording |
| parse failure | silently filled with 50 | recorded as missing |
| raw log | first 500 chars of the response | full answer distribution per item |

Why one item per call matters beyond fidelity to the paper: asking 44 items in
one response made each answer conditional on the previous ones, and the model
produced patterns rather than answers. The first respondent's twelve trust
items were 75, 70, 72 / 65, 60, 62 / 55, 50, 52 / 45, 40, 42 — a descending
staircase with an identical shape in each block of three. Separate calls remove
this by construction.

Diagnostics from the old run, for the record:
- 5,028 distinct response vectors across 9,000 rows. ~44% of respondents were
  exact duplicates of another respondent across all 25 outcomes; one vector
  repeated 108 times.
- Within-party spread was implausibly tight: control-condition trust was
  Democrat 79.7 (SD 5.4), Republican 54.5 (SD 3.3). Real subgroups on a 0–100
  trust item spread far wider. Tiny variance inflates apparent effect sizes.
- 202 rows had `inst_trust_mean` at exactly 50.0, indistinguishable from parse
  failures because both wrote 50.
- All 9,000 raw log records hit the 500-character truncation. Registration item
  K.2 requires complete unprocessed model responses for Tier 1, so that archive
  did not satisfy the requirement.

New files in `LLM_simulation/`:
- `stimuli.py` — parses the real stimulus texts; handles the state-adaptive arm
- `items.py` — all 44 scored items, wording verbatim from `codebook.csv`
- `prompt_qa.py` — the paper's QA / BIO / PORTRAY formats; letter-probability decoding
- `generate_outcomes.py` — the new pipeline (replaces `generate_outcomes_qwen.py`)
- `test_pipeline.py` — offline checks, no GPU needed
- `PROMPTING.md` — the prompt format and every setting behind it

`generate_outcomes_qwen.py` is left in place for comparison. It should not be run.

---

### Sample size: 9,000 → 36,000

The pool was at the benchmark floor exactly (1,000 control + 500 × 16). Now
4,000 control + 2,000 × 16.

The floor is the size of the *human* sample we are scored against. Our error
against the human result has three parts: how wrong our method is, noise in our
number, and noise in the human number. More synthetic respondents shrinks the
second. The third is fixed no matter what we do, which caps the gain.

For one intervention-vs-control effect on the 0–100 scale, assuming SD ≈ 20:

| our N per arm | our noise | combined noise | change |
|---|---|---|---|
| 500 (floor) | 1.10 | 1.55 | — |
| 1,000 | 0.77 | 1.34 | −13% |
| **2,000** | **0.55** | **1.22** | **−21%** |
| 5,000 | 0.35 | 1.15 | −26% |
| infinite | 0.00 | 1.10 | −29% |

4× the floor captures 21% of the 29% available. Method bias is probably worth
several points, so this is a second-order gain — but it is a real one, and the
README's "a huge pool cannot buy a better score" is loosely worded on this point.

Cost is manageable because each call generates exactly **one token**, and a
respondent's 44 prompts share a long identical prefix, so vLLM prefix caching
covers most of the input.

---

### DECISIONS — settings we chose that nobody specified

1. ~~**11 response options per 0–100 slider.**~~ Reversed the same day — see the
   entry above. Items are now asked as integers 0–100, as the survey asks them.
2. **All six demographics conditioned at once**, in survey order. The paper
   conditions on one attribute at a time. Tier 1 needs a whole person, so we
   stack them; whether the model handles six as well as one is untested.
3. **Home state drawn in proportion to resident population.** Only the
   state-adaptive arm reads it. Population-weighting matches what a US sample
   would produce; uniform-over-states would over-weight small states and shift
   the hazard mix. Current mix in that arm: 55.5% flood, 23.1% wildfire, 21.4%
   winter.
4. ~~**`--value_mode sample`.**~~ No longer applies — there is no distribution
   to sample from now that answers are generated directly.
5. **Temperature 1.0 — STILL OPEN.** Was "greedy, temperature 0" when answers
   came from logprobs. Now that answers are generated, temperature is the only
   source of between-respondent variation and 0 would collapse every identical
   profile onto one answer. 1.0 is the neutral placeholder, not a justified
   choice.
6. **Stimulus re-sent on all 44 calls**, placed between the demographics and the
   item, wrapped in the survey's own two TRANSITION paragraphs. The paper has no
   treatment at all, so this placement is ours.

---

### Bug fixes in `scripts/generate_profiles.py`

- `validate_pool()` hardcoded 9,000 rows and the 1,000/500 split, so the
  `--n-control` / `--n-intervention` flags the script advertised would have
  crashed validation. It now honours the requested sizes.
- Added a `state` column (see decision 3). It is a working column, dropped
  before the submission file is written.

---

### Still open

- Nothing is committed yet.
- `metadata.json` is still the organizers' template: team "example", model
  `gpt-4o-mini`, placeholder DOIs. Needs our real team id before `make clean`
  can name the prediction file.
- `registration.md` is untouched. Item J.1 asks how the final pipeline was
  chosen and what data it was validated against; we need an answer.
- `predictions/` still holds only the organizers' `example_*` files. Those are
  uniform random noise (`trust_post` × `belief_post` correlation = 0.009) and
  are useful only as a format template — there is no ground truth in this repo.
- No external validation set chosen yet.
