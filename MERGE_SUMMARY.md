# Merge of `rebuild-prompt-pipeline` — Completion Summary

**Date:** 2026-08-21  
**Branch:** `integrate-rebuild-pipeline` (not yet pushed to `origin`)  
**Status:** ✅ Complete, all validations passed

## What Was Merged

Your professor's `rebuild-prompt-pipeline` branch has been fully integrated. The merge brought:

1. **New Tier-1 pipeline** (`LLM_simulation/generate_outcomes.py`)
   - One LLM call per survey item (44 items) instead of one call for all 44 items
   - Eliminates the "descending staircase" answer patterns found in the old all-in-one prompt
   - Uses the professor's paper-based prompt format (QA/BIO/PORTRAY demographic conditioning)
   - Regex-constrained decoding for format validity

2. **Real intervention stimulus texts** (`LLM_simulation/stimuli.py`)
   - Parses all 16 intervention texts + 3 control fillers from `survey/questionnaire.txt`
   - No more placeholder text ("Condition: X, full intervention text should be provided")
   - Handles state-adaptive "Extreme weather predictions" arm with per-state case selection

3. **Item metadata and composite logic** (`LLM_simulation/items.py`)
   - All 44 items formally defined with their survey wording, scales, and metadata
   - `funding_perceptions` correctly marked as reverse-coded (`100 - raw_value`)
   - Composite outcome calculations moved into `generate_outcomes.py`

4. **Offline testing & dry-run mode** (`LLM_simulation/test_pipeline.py`, `stub_llm.py`)
   - `--dry_run` flag lets you test the pipeline without a GPU
   - Offline test suite found and fixed two bugs in the old run (undefined `chat_wrap`, wrong column position for `trust_multidimensional`)

5. **Profile pool scaled 4×** (9,000 → 36,000 respondents)
   - `scripts/generate_profiles.py` updated with new population-weighted `state` column
   - New `state` column drives the state-adaptive "Extreme weather predictions" arm
   - Profile defaults: `--n-control 4000`, `--n-intervention 2000` per arm

6. **Bug fixes**
   - `generate_profiles.py::validate_pool()` now honors `--n-control`/`--n-intervention` flags instead of hardcoding 9,000/1,000/500
   - `funding_perceptions` sign correction: `100 - raw` applied correctly
   - `trust_multidimensional` now at CSV column position 9 (was incorrectly at position 21)
   - `chat_wrap` function properly defined (was undefined in prior code)

7. **Documentation**
   - `CHANGELOG.md` — detailed record of all changes and the rationale behind them
   - `LLM_simulation/PROMPTING.md` — design decisions, settings, and their justifications

## What Was NOT Merged

- **`tier_2/` scripts removed** (per your decision this session): `generate_tier2_direct.py` and `README.md` deleted from the branch
- **`LLM_simulation/generate_outcomes_qwen.py` untouched** — the old pipeline is kept for reference but should not be used going forward
- **`LLM_simulation/questionnaire_parser.py` retained** — coexists with the new `stimuli.py` (both parse the questionnaire, but `stimuli.py` is now the primary consumer in the new pipeline)
- **Untracked files ignored**: `probing/representation_extract/` and `probing/replication_materials.zip` remain untracked (out of scope)

## Validation Results

All offline checks passed:

### 1. **Offline test suite** (`python LLM_simulation/test_pipeline.py`)
```
✓ Built 1,496 prompts across 34 profiles
✓ No placeholder stimulus text in prompts
✓ Extreme-weather arm renders cleanly (state=New York)
✓ Composites correct (funding reversed 30 → 70)
✓ Missing item propagates as missing, not 50
✓ Sliders asked as integers 0–100, as the survey does
✓ Answers parse; out-of-range and junk return None, not 50
✓ Decoding regexes admit the full scale and nothing else
✓ All 25 outcome columns produced
ALL CHECKS PASSED
```

### 2. **Dry-run smoke test** (50 profiles, no GPU)
```
Input:  50 profiles × 44 items = 2,200 LLM calls
Output: tier1_submission_20260821_152104.csv (50 rows × 33 columns)

CSV Structure:
  - profile_id, condition, [8 demographics], [25 outcomes]
  - trust_multidimensional at column 9 ✓
  - funding_perceptions at column 24 ✓
  - No placeholder-text leakage in raw_output_*.jsonl ✓

Example outcome ranges (dry-run random values):
  - trust_multidimensional:  mean=49.47, sd=8.67, range=[30.92, 65.00]
  - funding_perceptions:     mean=45.48, sd=30.25, range=[0.00, 97.00]
  - donation_ams:            mean=4.98, sd=3.37, range=[0.00, 10.00]
  - newsletter_signup:       mean=0.40, sd=0.49, range=[0.00, 1.00]

Parse failures (expected in dry-run with stub model):
  - 15 of 2,200 calls returned unparseable answers (0.682%) ✓
  - Correctly recorded as missing, not filled with 50
```

### 3. **Profile pool verification**
```
✓ 36,001 lines (header + 36,000 profiles)
✓ Schema includes new 'state' column (position 9)
  profile_id, condition, gender, age_band, race, education, income, party, state
```

## Key Settings in the New Pipeline

The professor left three settings flagged as **open decisions** (documented in `CHANGELOG.md`):

| Setting | Value | Status | Notes |
|---------|-------|--------|-------|
| **Temperature** | 1.0 | 🔴 Open | Only source of variation now that answers are generated per-item. She notes KAIST 2026 found temperature may not produce realistic variation; flagged for discussion. |
| **Enforce eager** | False (removed) | ⚠️  Note | Was `True` in old code. She removed it "to speed things up" but notes you may need it on Quest if CUDA graph capture fails. Watch for on first real GPU run. |
| **Stimulus re-sent** | All 44 calls | ✓ Final | Stimulus is included in every item's prompt. Relies on vLLM's prefix caching for efficiency. |
| **Control filler** | Random per respondent | ✓ Final | One of 3 fillers drawn per respondent (seeded). |
| **Extreme weather case** | State-adaptive | ✓ Final | Mapped from respondent's `state` column; default Case 4 if no state given. |
| **Newsletter response type** | Proportion (0–1) | ✓ Final | Interpreted as share who'd subscribe, matches Tier-2 spec. |

## Next Steps (For You)

1. **Review the changes:**
   - Read `CHANGELOG.md` for the full rationale
   - Read `LLM_simulation/PROMPTING.md` for the prompt design
   - Compare the new `generate_outcomes.py` against the old `generate_outcomes_qwen.py` if you want to understand what changed

2. **Real GPU smoke test (requires Quest):**
   ```bash
   python LLM_simulation/generate_outcomes.py --max_profiles 50 --output_dir /tmp/smoke
   ```
   This requires an actual GPU node. Check the vLLM logs for:
   - Prefix-cache hit rate (should be high since 44 prompts per profile share a long prefix)
   - Repeated-persona answer variability (monitor for mode collapse)

3. **Decide on temperature:**
   - Current default is 1.0 (her "neutral placeholder")
   - Consider the note about KAIST 2026's findings on temperature's limitations
   - You might want to use 0.0 for more deterministic central-tendency estimates, or stick with 1.0 for diversity
   - This decision should be made before the full run

4. **Full production run:**
   ```bash
   python LLM_simulation/generate_outcomes.py --seed <your_seed> --output_dir raw_data_deposit
   ```
   (36,000 profiles, ~1.6M calls, will take hours on Quest depending on your allocation)

5. **Merge into `main` when ready:**
   ```bash
   git checkout main
   git merge integrate-rebuild-pipeline
   git push origin main
   ```

6. **Rotate GitHub token:**
   The PAT you pasted (`ghp_3o...`) is now in plaintext git history. Go to https://github.com/settings/tokens and revoke it, then generate a new one for future use.

## Notes for Your Professor

If you want to brief your professor on what was applied:

> I've integrated your `rebuild-prompt-pipeline` branch into our working repo. The merge was clean — no conflicts beyond a trivial `.gitignore` rewrite. All your changes came through:
> 
> - ✅ New one-item-per-call prompt design + real stimulus texts
> - ✅ Funding_perceptions sign fix + funding_multidimensional column position fix
> - ✅ State-adaptive weather arm with per-respondent state assignment
> - ✅ 4× sample size (36k profiles)
> - ✅ Offline test suite (all 8 checks passing)
> - ✅ --dry_run mode (validated with 50 profiles, CSV structure correct)
> 
> The only things still flagged as open decisions (per your CHANGELOG):
> - Temperature = 1.0 (you noted it's a neutral placeholder, not justified)
> - enforce_eager removed from vLLM call (you noted it may be needed on Quest)
> 
> Ready for a real GPU smoke test on Quest whenever you want. Will watch the vLLM logs for prefix-cache hit rate and repeated-persona collapse.

---

**Branch location:** `/gpfs/projects/p32143/silicon-sample-submission` → `integrate-rebuild-pipeline`  
**Safe to review/iterate:** Yes, no uncommitted changes.  
**Ready to merge to main:** Yes, pending your review.
