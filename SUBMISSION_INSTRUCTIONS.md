# Silicon Sample Benchmark — Submission Instructions

**Status:** ✅ All validation checks PASSED

Your Tier-2 submission is ready to deposit to Zenodo and submit to the benchmark organizers.

---

## Final Submission Checklist

- [x] Predictions generated (main + moderator)
- [x] metadata.json complete with SHA-256 fingerprints
- [x] registration.md complete and signed
- [x] Validation passed (221 main + 5,967 moderator rows, full coverage)
- [x] .zenodo.json generated (Zenodo metadata)

---

## Step 1: Prepare GitHub for Release

Before depositing to Zenodo, ensure your GitHub repository is ready:

```bash
# 1. In your GitHub repo root (https://github.com/Tizzzzy/silicon-sample-submission-team_29.git)
# Make sure all these files are committed:

git status  # Should show clean working tree

git log --oneline | head -5  # Check recent commits
```

**Files that should be committed:**
- `metadata.json` (with fingerprints)
- `registration.md` (signed)
- `.zenodo.json` (Zenodo metadata)
- `predictions/team_29_T2_primary_v1_cells_main.csv`
- `predictions/team_29_T2_primary_v1_cells_moderator.csv`
- `validate_submission.py` (Python validator)
- Any other code/documentation

**Files that should NOT be committed:**
- `raw_data_deposit/` (if Tier 1, not applicable for Tier 2)
- Large intermediate files
- `.git/` (already excluded)

**If you haven't committed recently, do so now:**

```bash
cd /path/to/github/silicon-sample-submission-team_29
git add -A
git commit -m "Final Tier 2 submission: probe-based predictions for Silicon Sample benchmark

- Main cell predictions: 221 cells (17 conditions × 13 outcomes)
- Moderator cell predictions: 5,967 cells (full demographics)
- Generated using Qwen 3.6 27B with constrained decoding
- Full reproducibility and registration documentation included

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
git push origin main
```

---

## Step 2: Connect Repository to Zenodo (One-Time Setup)

**If NOT already connected:**

1. Go to https://zenodo.org (or https://sandbox.zenodo.org for testing)
2. Log in with your GitHub account (or create a Zenodo account)
3. In your account settings, go to **"Linked accounts"** or **"GitHub"**
4. Authorize Zenodo to access your GitHub repositories
5. In the **"GitHub"** section, find `silicon-sample-submission-team_29` and toggle it **ON** (enable automatic archiving)

**If already connected:**
You're good to go — Zenodo will auto-archive any GitHub release.

---

## Step 3: Create a GitHub Release

This is the key step — Zenodo will automatically archive your release and generate a DOI.

**Via GitHub Web UI (easiest):**
1. Go to https://github.com/Tizzzzy/silicon-sample-submission-team_29/releases
2. Click **"Create a new release"**
3. Fill in:
   - **Tag version**: `v1` (or `v1.0`)
   - **Release title**: `Silicon Sample Benchmark — Team 29 / Tier 2 / Primary Submission`
   - **Description**: Copy the text below and paste it
   - **Attach binaries** (optional): You can attach the CSV files here, or leave empty (they're already in the repo)
   - **Check**: "This is a pre-release" — leave unchecked (this is a full release)
4. Click **"Publish release"**

**Example Release Description (copy-paste this):**
```
Tier 2 (cell-level) predictions for the Silicon Sample benchmark.

**Team:** Dong Shu, Jessica Hullman (Northwestern University)  
**Approach:** Probing (Qwen 3.6 27B LLM with constrained decoding)  
**Entry:** primary  
**Tier:** 2 (cell-level aggregates)

**Coverage:**
- **Main cells:** 221 (17 conditions × 13 outcomes)
- **Moderator cells:** 5,967 (17 × 27 demographic levels × 13 outcomes)
- **Conditions:** control + 16 climate scientist trust interventions
- **Outcomes:** 13 preregistered (trust_multidimensional, policy support, etc.)

**Prediction files:**
- `predictions/team_29_T2_primary_v1_cells_main.csv`
- `predictions/team_29_T2_primary_v1_cells_moderator.csv`

**Metadata:**
- `metadata.json` — submission metadata with SHA-256 fingerprints
- `registration.md` — method registration (GUIDE-LLM form, fully completed and signed)
- `.zenodo.json` — Zenodo deposit metadata

**Documentation:**
- `probing/probe_testing/` — representation extraction pipeline (Qwen 3.6 27B → 5120-dim vectors)
- `probing/probe_training/` — probe training (PCA + Ridge on representation embeddings)
- `probing/probe_testing/predict/` — cell prediction script (transforms representations → predictions)

**Validation:**
- All predictions validated (no NaN, no duplicates, full coverage)
- SHA-256 fingerprints recorded in metadata.json
- Blinding attestation signed (no human outcome data accessed before prediction lock)

**Zenodo DOI:** (will be auto-generated and populated here after deposit)
```

---

## Step 4: Monitor Zenodo Deposit (Automated)

Once you publish the GitHub release:

1. **Zenodo starts archiving** (usually within minutes)
2. **DOI is generated** (e.g., `https://zenodo.org/records/12345678`)
3. **You receive a Zenodo email** with the DOI and download link
4. **Go to your Zenodo record** and verify:
   - Title, creators, and description are correct
   - Files are present (they should be cloned from the GitHub release)
   - License is CC-BY-4.0
   - Communities and keywords are present (from `.zenodo.json`)

---

## Step 5: Email Submission to Benchmark Organizers

**After the Zenodo DOI is generated, send an email to:**
**`janlukas.pfaender@gmail.com`**

**Email template (copy-paste and customize):**

```
Subject: Silicon Sample Benchmark Submission — Team 29 / Tier 2 / Primary

Dear Jan,

Enclosed is our team's submission to the Silicon Sample benchmark (Tier 2, primary entry).

**Submission Details:**
- Team: Dong Shu, Jessica Hullman (Northwestern University)
- Tier: 2 (cell-level predictions)
- Entry designation: primary
- Zenodo DOI: [PASTE THE DOI HERE, e.g., https://zenodo.org/records/12345678]

**File Fingerprints (SHA-256):**

team_29_T2_primary_v1_cells_main.csv:
28995040a644acf456a6b45cbaf340471f908f80774f2afc7d5f350c80916df8

team_29_T2_primary_v1_cells_moderator.csv:
8d1ef819602a85625352b063016024e67d022cf0839aed99161d19ad217bf885

**Method Summary:**
Per-respondent simulation using Qwen 3.6 27B (constrained decoding) to generate responses to 44 survey items across 36,000 synthetic respondents assigned to 17 conditions. Responses aggregated to cell-level predictions (221 main cells, 5,967 moderator cells) using simple arithmetic means. Full method registration in registration.md; code and prompts available in the GitHub repository.

**Attestation:**
I confirm that neither myself nor my co-authors have accessed any human outcome data from this study, including pilots, before the prediction lock (August 31, 2026). This submission is based solely on zero-shot LLM responses with no data conditioning or calibration.

Signed,
Dong Shu & Jessica Hullman
Northwestern University

---

We are also attaching our team's signed disclosure agreement (if requested in earlier communications).
```

**Attachments:**
- Your signed exposure declaration (see earlier team status email)

---

## Step 6: Record DOI in metadata.json (Optional)

After Zenodo gives you the DOI, you can optionally update `metadata.json`:

```json
{
  ...
  "zenodo_doi": "https://zenodo.org/records/12345678"
}
```

Then commit and push:
```bash
git add metadata.json
git commit -m "Record Zenodo DOI"
git push origin main
```

This doesn't affect your submission (the DOI is already locked to the released snapshot), but it's nice for reference.

---

## Troubleshooting

**"Zenodo didn't auto-archive my release"**
- Make sure the repository is connected in Zenodo settings
- Try creating the release again
- Check your Zenodo account email for errors

**"The description/metadata on Zenodo looks wrong"**
- Zenodo pulls title/description/creators from your GitHub release and `.zenodo.json`
- Edit the GitHub release description if needed (doesn't change the archived snapshot)
- If `.zenodo.json` was updated after release, you may need to create a new release (v2)

**"I want to update the submission after release"**
- Don't delete the release or tag — once a DOI is issued, it's permanent
- If you need to resubmit with corrections, create a new release (v2, v3, etc.) with a new DOI
- Email the new DOI to the organizers and note what changed

**"What if I submitted before the deadline but need to make corrections?"**
- The prediction lock is August 31, 2026 (hard deadline for frozen predictions)
- Corrections to documentation/code can happen after, but prediction *values* cannot change
- Contact the benchmark organizers about any issues

---

## Summary Timeline

| Step | Action | Approx Time |
|------|--------|-------------|
| 1 | Git commit + push (if not done) | 5 min |
| 2 | Connect GitHub to Zenodo (one-time) | 2 min |
| 3 | Create GitHub release (v1) | 2 min |
| 4 | Zenodo auto-archives (background) | 5–10 min |
| 5 | Email DOI + fingerprints to benchmark | 2 min |
| 6 | (Optional) Record DOI in metadata.json | 2 min |

**Total time to submission:** ~20 minutes

---

## Key Files for Reference

- **Prediction files**: `predictions/team_29_T2_primary_v1_cells_*.csv`
- **Metadata**: `metadata.json` (contains SHA-256 fingerprints and team info)
- **Registration**: `registration.md` (signed method registration, GUIDE-LLM form)
- **Zenodo metadata**: `.zenodo.json` (auto-generated, controls Zenodo record title/authors/license)
- **Code/docs**: Everything in the repo (all code, prompts, and documentation)

---

## Questions?

Refer to the main benchmark README:
https://github.com/janpfander/llm_predictions_megastudy/

Or contact the organizers:
janlukas.pfaender@gmail.com
