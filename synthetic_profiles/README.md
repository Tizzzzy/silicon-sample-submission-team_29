# Synthetic Profile Pool for Silicon Sample Benchmark

## Overview

This directory contains the **synthetic respondent profile pool** generated for the Silicon Sample Benchmark submission. The profiles are demographic data for 9,000 synthetic respondents distributed across 17 conditions (1 control + 16 message interventions).

## What's in Here

### `profiles_pool.csv`

A CSV file with 9,000 rows (one per synthetic respondent) and 8 columns:

| Column | Type | Values | Notes |
|--------|------|--------|-------|
| `profile_id` | string | p00000, p00001, ... p08999 | Unique identifier, zero-padded 5-digit format |
| `condition` | string | "control", "Corporate reliance", ... | 17 values: control + 16 intervention titles |
| `gender` | string | Male, Female, Other | 3 levels |
| `age_band` | string | 18-29, 30-44, 45-59, 60+ | 4 bands (Census-based quotas) |
| `race` | string | White / Caucasian, Black / African American, Hispanic / Latino, Asian / Asian American, Other | 5 levels (Census-based quotas) |
| `education` | string | 6 levels from "Less than high school" to "Doctorate degree / Ph.D." | Survey instrument exact wording |
| `income` | string | 5 levels from "Less than $30,000" to "$168,000 or more" | Survey instrument exact wording |
| `party` | string | Republican, Democrat, Independent, Other | 4 levels |

## Data Distribution

Generated with **seed 2026** (reproducible):

- **Total profiles:** 9,000
  - **Control:** 1,000 (2,000 in the human study, but 500/condition is the Tier-1 submission minimum)
  - **Each intervention (×16):** 500 profiles
  
- **Demographic targets** (independent draws per condition):
  - `age_band` and `race`: **exact Census quotas from benchmark preregistration** (load-bearing for scoring)
  - `gender`, `education`, `income`, `party`: documented US national priors (Census ACS, Pew/Gallup)

All moderators are drawn **independently** within each condition (no joint correlation structure). This means every condition's demographic distribution approximates the same national marginals, mirroring the real study's independent per-arm quota sampling.

## What This Is and Isn't

✓ **This is:** A profile pool with condition assignment and 6 demographic moderators — everything needed to assign to a stimulus and construct prompt instructions for an LLM.

✗ **This is NOT:** A complete Tier-1 submission file. Missing columns:
  - 13 **outcome variables** (trust_multidimensional, trust_post, distrust_post, ... donation_ams, newsletter_signup)
  - 12 **trust sub-items** (trust_competence_1..3, trust_integrity_1..3, trust_benevolence_1..3, trust_openness_1..3)

These must be generated in a **separate pipeline step** (LLM calls per profile + intervention message, then composite calculations).

## How to Use This Pool

### Step 1: Load the profiles

```python
import pandas as pd
df = pd.read_csv("synthetic_profiles/profiles_pool.csv")
# df has 9,000 rows and 8 columns
```

### Step 2: Add intervention stimulus text

For each row, retrieve the full intervention message from `survey/` or your stimulus library and assign to that profile:

```python
# Pseudo-code
for _, row in df.iterrows():
    profile_id = row["profile_id"]
    condition = row["condition"]
    stimulus = get_intervention_text(condition)
    
    # Next: call LLM to generate outcomes
```

### Step 3: Simulate outcomes via LLM

Call an LLM with a prompt that includes:
- The demographic profile (gender, age_band, race, education, income, party)
- The condition/stimulus text
- The survey questions (from `survey/questionnaire.txt` or `survey/survey.json`)

The LLM should generate responses to all 13 outcomes + 12 trust sub-items for that profile.

### Step 4: Assemble final Tier-1 submission CSV

Combine `profiles_pool.csv` with the simulated outcomes to create the final submission file with all 25 columns (8 demographic + 12 trust sub-items + 13 outcomes).

## Generator Script

**Location:** `scripts/generate_profiles.py`

Generated the pool with:

```bash
python scripts/generate_profiles.py --seed 2026 --out synthetic_profiles/profiles_pool.csv
```

### Options

```
--seed SEED                 Random seed for reproducibility (default: 2026)
--out OUTPUT_PATH          Output CSV file (default: synthetic_profiles/profiles_pool.csv)
--n-control N_CONTROL      Number of control profiles (default: 1,000)
--n-intervention N_INTERV  Number per intervention (default: 500; total = 16×500 + 1,000 = 9,000)
```

### Scaling Example

To generate a full 18,000-profile pool (2:1 human ratio):

```bash
python scripts/generate_profiles.py --seed 2026 --n-control 2000 --n-intervention 1000 --out synthetic_profiles/profiles_pool_18k.csv
```

### Reproducibility

Same seed always produces the same profiles (same order, same demographics):

```bash
python scripts/generate_profiles.py --seed 2026 --out /tmp/test1.csv
python scripts/generate_profiles.py --seed 2026 --out /tmp/test2.csv
# /tmp/test1.csv == /tmp/test2.csv ✓
```

Different seed produces different demographics:

```bash
python scripts/generate_profiles.py --seed 42 --out /tmp/test_seed_42.csv
# Different demographics, same structure and counts
```

## Demographic Source Attribution (for `registration.md`)

For section **D.1 (Profile Source)** in `registration.md`, cite:

- **age_band, race:** Exact Census 2024 quotas from benchmark preregistration ([janpfander.github.io/llm_predictions_megastudy/preregistration_benchmark.html](https://janpfander.github.io/llm_predictions_megastudy/preregistration_benchmark.html))
- **gender:** ~49% Male / ~49% Female / ~2% Other (survey-panel estimate)
- **education:** Census ACS distribution, adults 25+ (US Census Bureau, American Community Survey)
- **income:** Census ACS household income distribution, rebucketed to survey scale (US Census Bureau)
- **party:** Gallup aggregate party identification estimate (Gallup, recent surveys)

All moderators drawn independently via `numpy.random.default_rng().choice()` with probabilities set to these targets. No joint correlation structure imposed.

## Validation

The generated pool is validated for:
- ✓ Exact 9,000 rows
- ✓ Per-condition counts: control=1,000, each intervention=500
- ✓ All moderator levels match exact strings in `scripts/lib/submission_spec.R`
- ✓ No missing values
- ✓ Unique `profile_id` values
- ✓ Demographic distributions ±1% of targets overall, ~5% per-condition

Run the generator with verbose output to see a per-moderator distribution summary.

## Next Steps

1. **Load** `profiles_pool.csv` into your LLM outcome-simulation pipeline
2. **For each profile:** assemble prompt with demographics + condition stimulus + survey questions
3. **Call LLM** to generate outcome values (13 outcomes + 12 trust sub-items)
4. **Assemble** final Tier-1 submission CSV with all 25 columns
5. **Validate** via `make check` from the repo root — confirms structure, coverage, and value ranges
6. **Register** via `registration.md` and Zenodo before August 31, 2026

## Questions?

Refer to:
- **Benchmark preregistration:** [janpfander.github.io/llm_predictions_megastudy/preregistration_benchmark.html](https://janpfander.github.io/llm_predictions_megastudy/preregistration_benchmark.html)
- **FAQ:** [janpfander.github.io/llm_predictions_megastudy/faq.html](https://janpfander.github.io/llm_predictions_megastudy/faq.html)
- **Questionnaire:** [janpfander.github.io/llm_predictions_megastudy/questionnaire.html](https://janpfander.github.io/llm_predictions_megastudy/questionnaire.html)
- **Generator source:** `scripts/generate_profiles.py`
