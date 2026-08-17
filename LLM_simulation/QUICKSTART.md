# LLM Outcome Generation — Quick Start Guide

## TL;DR

### Test on 100 profiles (5 minutes)
```bash
cd /gpfs/projects/p32143/silicon-sample-submission/LLM_simulation
python generate_outcomes_qwen.py \
  --max_profiles 100 \
  --batch_size 8 \
  --output_dir ./test_output_100
```

### Run full pipeline (9,000 profiles)
```bash
python generate_outcomes_qwen.py \
  --batch_size 32 \
  --temperature 0.5 \
  --output_dir ../raw_data_deposit
```

### Process the generated CSV
```bash
cd ..
Rscript scripts/clean.R LLM_simulation/<output_csv_name>
make check
```

---

## Step-by-Step for First Run

### 1. Verify everything is set up

```bash
# Check profile pool exists
ls -lh ../synthetic_profiles/profiles_pool.csv

# Check model is available
ls -lh /projects/p32143/cache/qwen36_27b

# Check Python packages
python -c "import pandas, vllm; print('✓ All packages available')"
```

### 2. Test with small batch (safety check)

```bash
# Create test output directory
mkdir -p test_output_10

# Run on just 10 profiles
python generate_outcomes_qwen.py \
  --profile_pool ../synthetic_profiles/profiles_pool.csv \
  --model_path /projects/p32143/cache/qwen36_27b \
  --batch_size 4 \
  --max_profiles 10 \
  --temperature 0.7 \
  --output_dir ./test_output_10

# This should take ~2-3 minutes
# Monitor GPU: nvidia-smi
```

### 3. Check test outputs

```bash
# List generated files
ls -lh test_output_10/

# Check final CSV shape and sample rows
python << 'EOF'
import pandas as pd
df = pd.read_csv("test_output_10/tier1_submission_*.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(df.head(2))
print("\nOutcome stats:")
print(df[["trust_multidimensional", "donation_ams", "newsletter_signup"]].describe())
EOF

# Check raw LLM outputs
tail -2 test_output_10/raw_output_*.jsonl | python -m json.tool
```

### 4. Once confident, run full 9,000 profiles

```bash
# Ensure output directory exists
mkdir -p ../raw_data_deposit

# Run full pipeline
python generate_outcomes_qwen.py \
  --profile_pool ../synthetic_profiles/profiles_pool.csv \
  --model_path /projects/p32143/cache/qwen36_27b \
  --batch_size 32 \
  --temperature 0.5 \
  --output_dir ../raw_data_deposit \
  --seed 2026

# Monitor progress in another terminal:
# tail -f ../raw_data_deposit/generate_outcomes_*.log
```

### 5. Post-process and validate

```bash
# Find the generated CSV
CSV_FILE=$(ls -t ../raw_data_deposit/tier1_submission_*.csv | head -1)
echo "Processing: $CSV_FILE"

# Clean with R (adds calculated composites, recodes some variables)
cd ..
Rscript scripts/clean.R "$CSV_FILE"

# Validate structure and values
make check
```

---

## Common Command Patterns

### Quick test (100 profiles, all default settings)
```bash
python generate_outcomes_qwen.py --max_profiles 100
```

### Faster processing (larger batch size)
```bash
python generate_outcomes_qwen.py --batch_size 64 --temperature 0.3
```

### More creative outputs (higher temperature)
```bash
python generate_outcomes_qwen.py --batch_size 16 --temperature 0.9
```

### Reproducible (same seed, same outputs)
```bash
python generate_outcomes_qwen.py --seed 42
```

### Custom output directory
```bash
python generate_outcomes_qwen.py --output_dir /tmp/my_outcomes
```

### Limit to specific profiles (for debugging)
```bash
python generate_outcomes_qwen.py --max_profiles 50 --batch_size 8
```

---

## What to Monitor

### During execution

**Open a new terminal and watch:**
```bash
# Real-time log tail
tail -f raw_data_deposit/generate_outcomes_*.log

# GPU usage
watch -n 2 'nvidia-smi | head -20'

# Model loading
ps aux | grep python
```

### Expected metrics

- **Speed:** ~10-50 profiles/minute per GPU (depends on batch size, temperature)
- **GPU memory:** ~20-30 GB with tensor_parallel_size=2
- **CPU:** Moderate (mostly disk I/O and batching)

### Logs to check

```bash
# Summary at the end
tail -50 raw_data_deposit/generate_outcomes_*.log

# Error details
grep -i "error\|warning" raw_data_deposit/generate_outcomes_*.log | head -20

# Per-profile parsing issues
grep "Failed to parse" raw_data_deposit/generate_outcomes_*.log | wc -l
```

---

## Expected Output Files

After running, you should see in `raw_data_deposit/` or your `--output_dir`:

```
raw_data_deposit/
├── raw_output_20260817_120000.jsonl    ← Raw LLM responses (for debugging)
├── tier1_submission_20260817_120000.csv ← Tier-1 ready CSV (main output)
└── generate_outcomes_20260817_120000.log ← Detailed log
```

### File sizes (rough estimates for 9,000 profiles)

- `raw_output_*.jsonl`: ~50-100 MB (depends on response length)
- `tier1_submission_*.csv`: ~15-20 MB
- `generate_outcomes_*.log`: ~5-10 MB

---

## Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| Script hangs at "Initializing vLLM" | Reduce `--tensor_parallel_size 1` or check `nvidia-smi` |
| OOM (out of memory) error | Use `--batch_size 8` instead of 32, or `--gpu_memory_utilization 0.65` |
| Parsing warnings | Increase `--max_tokens 2048`, try `--temperature 0.3` |
| Outputs all zeros/garbage | Check raw_output JSONL, may need prompt adjustment |
| CSV file too large / slow to load | Subset with `--max_profiles 1000` to test |
| Different outputs each run | This is normal; use `--seed 42` to fix seed |

---

## Registration.md Sections to Fill

After successful generation, add to `registration.md`:

**Section D.1 (Profile Source):**
> Profiles sourced from scripts/generate_profiles.py (seed 2026).
> Demographics follow Census quotas for age/race + national priors for gender/education/income/party.
> See synthetic_profiles/README.md for full sources.

**Section D.2 (Profile Verbalization):**
> Demographic profiles passed as context to LLM prompts.
> Format: "Respondent: [Gender], [Age band] years old, [Race], [Education], income [Income], [Party]"

**Section D.3 (Assignment & Weighting):**
> 9,000 synthetic respondents (1,000 control + 500 per intervention).
> Each profile assigned to exactly one condition. No reuse across conditions.

**Section K (Reproducibility):**
> Generated with:
> - Profile pool: `python scripts/generate_profiles.py --seed 2026`
> - Outcomes: `python LLM_simulation/generate_outcomes_qwen.py --seed 2026 --batch_size 32`
> - Model: Qwen 3.6-27B via vLLM (tensor_parallel_size=2)
> Code: [link to GitHub repo release]

---

## Estimated Time & Cost

### For 9,000 profiles:

| Setting | Speed | Est. Time | Notes |
|---------|-------|-----------|-------|
| `batch_size=8` | ~10 profiles/min | 15 hours | Safe, conservative |
| `batch_size=32` | ~30 profiles/min | 5 hours | Recommended |
| `batch_size=64` | ~40 profiles/min | 3-4 hours | Max speed, high GPU memory |

**GPU time:** ~6-15 GPU-hours (2× GPUs × 3-8 wall-clock hours)

**Cost:** Depends on your cluster/cloud provider. Local GPU costs nothing if already available.

---

## Next Steps After Generation

1. ✓ Run `python generate_outcomes_qwen.py` → generates CSV
2. ✓ Check `raw_output_*.jsonl` for quality
3. → Run `Rscript scripts/clean.R <CSV>`
4. → Run `make check`
5. → Fill `registration.md`
6. → Create GitHub release + Zenodo deposit
7. → Send DOI by August 31, 2026

---

## Quick Stats Check Script

After generating, run this to verify outputs:

```python
import pandas as pd
import glob

# Load the latest tier1 CSV
csv_file = max(glob.glob("tier1_submission_*.csv"), key=lambda x: x)
df = pd.read_csv(csv_file)

print(f"✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"✓ Required columns present: {len(df.columns) == 25}")
print(f"✓ No missing values: {df.isnull().sum().sum() == 0}")

# Check per-condition
for cond in df["condition"].unique():
    count = (df["condition"] == cond).sum()
    print(f"  {cond:35s}: {count:4d} profiles")

# Outcome ranges
print("\nOutcome value ranges (should be in valid bounds):")
for col in ["trust_multidimensional", "trust_post", "donation_ams", "newsletter_signup"]:
    values = df[col]
    print(f"  {col:30s}: [{values.min():6.1f}, {values.max():6.1f}]")

print("\n✓ All checks passed!")
```

---

## Get Help

1. **Script errors:** Check `generate_outcomes_*.log`
2. **Parsing issues:** Review `raw_output_*.jsonl` (sample a few records)
3. **vLLM issues:** See `README.md` troubleshooting section
4. **Benchmark questions:** Contact janlukas.pfaender@gmail.com
