# LLM Outcome Generation for Silicon Sample Benchmark

## Overview

This directory contains scripts to generate all 13 outcomes + 12 trust sub-items for synthetic respondent profiles using a local Qwen LLM instance via vLLM.

**Pipeline:**
1. Load the demographic profile pool (`synthetic_profiles/profiles_pool.csv`)
2. For each profile, construct a survey prompt (demographics + intervention + questions)
3. Batch prompts and call Qwen LLM via vLLM for inference
4. Parse LLM outputs to extract outcome values (0-100 scales, 0-10 donation, 0-1 newsletter)
5. Compute composite outcomes (means of sub-items)
6. Assemble final Tier-1 submission CSV with all 25 required columns

## Files

### Scripts

- **`Qwen_example.py`** — Minimal working example using vLLM with Qwen
  - Shows how to load the model, format prompts, batch generate, and extract outputs
  - Good reference for troubleshooting vLLM initialization

- **`generate_outcomes_qwen.py`** — Full outcome generation pipeline (THIS IS THE MAIN SCRIPT)
  - Loads profile pool
  - Constructs survey prompts for each profile
  - Batch inference with vLLM
  - Parses outputs and computes composites
  - Saves raw outputs + final Tier-1 CSV

### Configuration

- **`.env`** (optional) — Environment variables for model path, GPU settings, etc.

## Quick Start

### 1. Verify the model is available

```bash
ls -lh /projects/p32143/cache/qwen36_27b
# Should show the model directory
```

### 2. Run on a small test batch (10 profiles)

```bash
python generate_outcomes_qwen.py \
  --profile_pool ../synthetic_profiles/profiles_pool.csv \
  --model_path /projects/p32143/cache/qwen36_27b \
  --batch_size 4 \
  --max_profiles 10 \
  --output_dir ./test_output \
  --seed 2026
```

### 3. Check the outputs

```bash
ls -lh test_output/
# You should see:
#   - raw_output_<timestamp>.jsonl (raw LLM responses)
#   - tier1_submission_<timestamp>.csv (Tier-1 ready file)
#   - generate_outcomes_<timestamp>.log (detailed log)

head test_output/tier1_submission_*.csv
tail test_output/raw_output_*.jsonl | python -m json.tool
```

### 4. Run on the full profile pool (9,000 profiles)

```bash
python generate_outcomes_qwen.py \
  --profile_pool ../synthetic_profiles/profiles_pool.csv \
  --model_path /projects/p32143/cache/qwen36_27b \
  --batch_size 16 \
  --temperature 0.5 \
  --top_p 0.95 \
  --output_dir ../raw_data_deposit \
  --seed 2026
```

This will:
- Process all 9,000 profiles
- Save raw outputs to `raw_output_<timestamp>.jsonl`
- Save final Tier-1 CSV to `tier1_submission_<timestamp>.csv` in `raw_data_deposit/`
- Generate a detailed log file

## Command-Line Options

### Model Configuration

```
--model_path PATH              Path to Qwen model (default: /projects/p32143/cache/qwen36_27b)
--tensor_parallel_size N       Number of GPUs for tensor parallelism (default: 2)
--gpu_memory_utilization FRAC  GPU memory fraction to use (default: 0.85)
```

### Sampling Parameters

```
--batch_size N                 Batch size for vLLM (default: 8; try 16-32 for faster processing)
--max_tokens N                 Max tokens to generate per sample (default: 1024)
--temperature TEMP             Sampling temperature 0-1 (default: 0.7)
--top_p P                      Top-p nucleus sampling (default: 1.0)
--seed SEED                    Random seed for reproducibility (default: None)
```

### I/O and Debugging

```
--profile_pool PATH            Input profile pool CSV (default: ../synthetic_profiles/profiles_pool.csv)
--output_dir PATH              Output directory (default: raw_data_deposit/)
--max_profiles N               Limit to first N profiles for testing (default: None, process all)
```

## Output Files

### `raw_output_<timestamp>.jsonl`

JSONL file with raw LLM outputs for inspection and debugging. Each line is a JSON record:

```json
{
  "profile_id": "p00000",
  "condition": "control",
  "raw_llm_output": "trust_competence_1: 65\ntrust_competence_2: 70\n...",
  "parsed_outcomes": {
    "trust_competence_1": 65,
    "trust_competence_2": 70,
    "..._mean": 67.5,
    "trust_multidimensional": 65.8,
    ...
  }
}
```

Use this to debug parsing issues or inspect what the LLM generated.

### `tier1_submission_<timestamp>.csv`

Tier-1 submission-ready CSV with all 25 required columns:
- `profile_id`, `condition`
- 6 demographic moderators (gender, age_band, race, education, income, party)
- 12 trust sub-items (trust_competence_1..3, trust_integrity_1..3, etc.)
- 13 outcomes (trust_multidimensional, trust_post, distrust_post, ..., donation_ams, newsletter_signup)

Ready to be processed by `scripts/clean.R` or directly validated with `make check`.

### `generate_outcomes_<timestamp>.log`

Detailed log file with:
- Configuration summary
- Processing progress (batch #, # profiles processed)
- Parsing warnings/errors
- Summary statistics

## Troubleshooting

### vLLM initialization hangs or OOM errors

**Symptom:** Script gets stuck at "Initializing vLLM engine..."

**Solutions:**
1. Reduce `--tensor_parallel_size` (e.g., 1 instead of 2)
2. Lower `--gpu_memory_utilization` (e.g., 0.7 instead of 0.85)
3. Check GPU availability: `nvidia-smi`
4. Try the example script first: `python Qwen_example.py`

### LLM output parsing failures

**Symptom:** "No key-value pairs found in output" warnings

**Causes:**
- Prompt structure not clear enough for LLM
- LLM generating text instead of structured key-value pairs

**Solutions:**
1. Increase `--temperature` to 0.3 for more deterministic output
2. Increase `--max_tokens` if the response is being cut off
3. Review raw outputs in the JSONL file
4. Adjust the prompt in `construct_survey_prompt()` to be more explicit

### Outcomes out of range

**Symptom:** Warnings like "trust_competence_1 = 110 out of range [0, 100]"

**Causes:**
- LLM generated invalid values

**Solutions:**
1. Values are automatically clipped to valid ranges
2. If frequent, adjust temperature or prompt clarity
3. Check raw outputs to understand what the LLM is generating

## Performance Tips

### Batch Size

- Larger batches are faster: try `--batch_size 32` if GPU memory allows
- Rule of thumb: `batch_size = (GPU_memory_GB × 0.85) / 5`

### Temperature and Sampling

- Lower temperature (0.3-0.5): faster, more deterministic
- Higher temperature (0.7-1.0): slower, more variety (may be needed for realistic human variance)

### Reproducibility

- Use `--seed 2026` to make outputs reproducible
- Must disclose the seed in `registration.md` section K

## Outcome Variables

### 25 Required Columns

```
profile_id                      Unique identifier (p00000-p08999)
condition                       Control or intervention title

DEMOGRAPHICS (6):
gender                          Male | Female | Other
age_band                        18-29 | 30-44 | 45-59 | 60+
race                            White / Caucasian | Black / African American | ... | Other
education                       Less than high school | ... | Doctorate degree / Ph.D.
income                          Less than $30,000 | ... | $168,000 or more
party                           Republican | Democrat | Independent | Other

TRUST SUB-ITEMS (12):
trust_competence_1..3           How competent/incompetent are climate scientists?
trust_integrity_1..3            How honest/dishonest are climate scientists?
trust_benevolence_1..3          How concerned/unconcerned about wellbeing?
trust_openness_1..3             How open to feedback/non-responsive?

OUTCOMES (13):
trust_multidimensional          PRIMARY OUTCOME: mean of 4 subscales (0-100)
trust_post                      Single-item trust question (0-100)
distrust_post                   Single-item distrust question (0-100)
funding_perceptions             Government funding too much/too little (0-100)
policy_role_mean                Mean of 4 items on scientists' policy role (0-100)
inst_trust_mean                 Mean of 5 institutional trust items (0-100)
belief_post                     "Human activities cause climate change" (0-100)
concern_mean                    Mean of 3 concern items (0-100)
policy_general                  "Government should do more to reduce warming" (0-100)
policy_specific_mean            Mean of 7 specific climate policies (0-100)
behavior_mean                   Mean of 6 climate-friendly behaviors (0-100)
donation_ams                    Donation to American Meteorological Society ($0-$10)
newsletter_signup               "Talking Climate" newsletter (0=No, 1=Yes)
```

## Integration with Benchmark Submission

### After outcome generation:

1. **Raw data processing:**
   ```bash
   cd ..
   Rscript scripts/clean.R LLM_simulation/tier1_submission_<timestamp>.csv
   ```

2. **Validation:**
   ```bash
   make check
   ```

3. **Registration & Zenodo:**
   - Fill `registration.md` with:
     - Section D.1: Profile source (demographics, distributions, seed)
     - Section D.2: Profile verbalization (we pass as text context to LLM)
     - Section D.3: Assignment & weighting
     - Section K: Reproducibility (seed, code, vLLM config)

4. **Submit:**
   - Create GitHub release with all files
   - Upload to Zenodo
   - Send DOI to `janlukas.pfaender@gmail.com` by August 31, 2026

## Example Outcome Distribution Check

After running, you can verify the distributions look reasonable:

```python
import pandas as pd

df = pd.read_csv("raw_data_deposit/tier1_submission_*.csv")

# Check outcome distributions
print(df[["trust_multidimensional", "trust_post", "policy_general"]].describe())

# Check by condition
print(df.groupby("condition")["trust_multidimensional"].mean().sort_values())

# Check that demographics are present
print(df[["gender", "age_band", "race"]].value_counts())
```

## Next Steps

1. **Run test batch:** `python generate_outcomes_qwen.py --max_profiles 100`
2. **Review outputs:** Check raw_output JSONL and tier1_submission CSV
3. **Adjust temperature/prompt if needed:** Based on output quality
4. **Run full pipeline:** `python generate_outcomes_qwen.py` (all 9,000 profiles)
5. **Process & validate:** `Rscript ../scripts/clean.R ...` → `make check`
6. **Register:** Fill registration.md with method details
7. **Submit by August 31, 2026** ✓

## Questions?

Refer to:
- **Benchmark details:** https://janpfander.github.io/llm_predictions_megastudy
- **Preregistration:** https://janpfander.github.io/llm_predictions_megastudy/preregistration_benchmark.html
- **Questionnaire:** https://janpfander.github.io/llm_predictions_megastudy/questionnaire.html
- **vLLM docs:** https://docs.vllm.ai/en/latest/
- **Qwen docs:** https://qwenlm.github.io/blog/qwen-intro/
