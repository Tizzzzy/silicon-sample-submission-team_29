#!/usr/bin/env python3
"""
Build grouped-level probe dataset from individual-level representations (v4).

Memory-efficient version using custom streaming JSON array parser.
"""

import json
import numpy as np
import re
from collections import defaultdict
from itertools import combinations
import time
import warnings
import sys

warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "../representation_extract/extracted_representations_v4.json"
OUTPUT_NPZ = "probe_dataset.npz"
OUTPUT_METADATA = "probe_dataset_metadata.json"
DIMENSIONS = ["education", "gender", "party", "political_knowledge"]
MIN_N = 3
REP_DIM = 5120

print("="*80)
print("Build Probe Dataset from Individual-Level Representations (v4)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Input file: {INPUT_FILE}")
print(f"  Output NPZ: {OUTPUT_NPZ}")
print(f"  Output metadata: {OUTPUT_METADATA}")
print(f"  Min group size: {MIN_N}")
print(f"  Dimensions: {DIMENSIONS}")

# ============================================================================
# Parse group_id using regex
# ============================================================================

GROUP_ID_RE = re.compile(
    r"^education_(?P<education>.+)_gender_(?P<gender>.+)"
    r"_party_(?P<party>.+)_political_knowledge_(?P<political_knowledge>.+)$"
)

def parse_group_id(group_id):
    """Parse group_id string into demographic dimensions dict."""
    m = GROUP_ID_RE.match(group_id)
    if not m:
        raise ValueError(f"group_id does not match expected pattern: {group_id!r}")
    dims = m.groupdict()
    # Round-trip self-check (defensive)
    rebuilt = "_".join(f"{d}_{dims[d]}" for d in DIMENSIONS)
    assert rebuilt == group_id, f"round-trip mismatch: {rebuilt!r} != {group_id!r}"
    return dims

# ============================================================================
# Generate all non-empty dimension subsets
# ============================================================================

def generate_dimension_subsets(dimensions=DIMENSIONS):
    """Generate all 15 non-empty subsets of dimensions."""
    subsets = []
    for r in range(1, len(dimensions) + 1):
        for combo in combinations(dimensions, r):
            subsets.append(combo)
    return subsets

subsets = generate_dimension_subsets()
assert len(subsets) == 15, f"Expected 15 subsets, got {len(subsets)}"

# ============================================================================
# Custom streaming JSON array parser (memory-efficient)
# ============================================================================

def stream_json_array(file_path, array_key='representations'):
    """
    Stream-parse a JSON file with structure:
    {"metadata": {...}, "array_key": [...items...]}

    Yields items from the array one at a time.
    Also returns metadata dict (extracted from file start).
    """
    with open(file_path, 'r') as f:
        # Read until we find the array start
        buffer = ""
        in_metadata = False
        metadata_text = ""
        found_array = False
        depth = 0
        escape = False

        for chunk_idx, chunk in enumerate(iter(lambda: f.read(65536), '')):
            buffer += chunk

            # Simple state machine: find "metadata": {...}, then array_key: [
            if not found_array:
                # Look for array start
                marker = f'"{array_key}": ['
                idx = buffer.find(marker)
                if idx != -1:
                    # Found array start
                    metadata_text = buffer[:idx]
                    remainder = buffer[idx + len(marker):]
                    found_array = True

                    # Extract metadata by parsing JSON up to the array
                    try:
                        meta = json.loads('{' + metadata_text.split('{', 1)[1].rsplit('}', 1)[0] + '}')
                        yield ('metadata', meta)
                    except:
                        yield ('metadata', {})

                    # Process remainder
                    buffer = remainder
                else:
                    continue

            if found_array:
                # Now parse items from array
                # Split by top-level },{} patterns
                while True:
                    # Try to extract one complete object
                    obj_start = 0
                    depth = 0
                    escape = False
                    in_string = False
                    idx = 0

                    for idx, c in enumerate(buffer):
                        if escape:
                            escape = False
                            continue
                        if c == '\\':
                            escape = True
                            continue
                        if c == '"':
                            in_string = not in_string
                            continue
                        if in_string:
                            continue

                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0 and idx > 0:
                                # Found end of object
                                obj_text = buffer[obj_start:idx+1]
                                try:
                                    obj = json.loads(obj_text)
                                    yield ('item', obj)
                                except Exception as e:
                                    print(f"Skipping malformed JSON object: {e}", file=sys.stderr)
                                # Look for next object
                                buffer = buffer[idx+1:].lstrip(',\n\t ')
                                obj_start = 0
                                depth = 0
                                break
                    else:
                        # Need more data
                        break

        # Process any remaining buffer
        if found_array and buffer.strip() and buffer.strip() != ']':
            buffer = buffer.rstrip('], \n\t')
            if buffer:
                try:
                    obj = json.loads(buffer)
                    yield ('item', obj)
                except:
                    pass

# ============================================================================
# PASS 1: Load JSON streaming and build finest buckets
# ============================================================================

print(f"\nPass 1: Loading {INPUT_FILE} and building finest buckets...")
t0 = time.time()

finest = {}
gt_by_group = {}
entry_count = 0
metadata = {}

for key, data in stream_json_array(INPUT_FILE, 'representations'):
    if key == 'metadata':
        metadata = data
        print(f"  Loaded metadata: {metadata.get('n_total_probes', '?')} probes")
        continue

    entry = data
    entry_count += 1

    if entry_count % 5000 == 0:
        print(f"  Processing entry {entry_count}...", flush=True)

    dims = parse_group_id(entry["group_id"])
    fine_key = (
        entry["question_id"],
        entry["condition"],
        dims["education"],
        dims["gender"],
        dims["party"],
        dims["political_knowledge"],
    )

    vec = np.asarray(entry["representation"], dtype=np.float64)

    b = finest.get(fine_key)
    if b is None:
        b = {
            "sum_vec": np.zeros(REP_DIM, dtype=np.float64),
            "sum_opinion": 0.0,
            "n": 0,
            "question_label": entry["question_label"],
        }
        finest[fine_key] = b

    b["sum_vec"] += vec
    b["sum_opinion"] += entry["individual_opinion"]
    b["n"] += 1

    # Store ground truth for cross-validation
    gid = entry["group_id"]
    qid = entry["question_id"]
    gt_key = (gid, qid)
    if gt_key not in gt_by_group:
        gt_by_group[gt_key] = {
            'stored_mean': entry['ground_truth_mean'],
            'stored_n': entry['n_respondents_in_group'],
        }

print(f"Pass 1 complete in {time.time() - t0:.1f}s")
print(f"  Built {len(finest)} finest buckets from {entry_count} entries")

# Sanity check 1
total_n = sum(b["n"] for b in finest.values())
print(f"\nSanity Check 1: Entry count conservation")
print(f"  Sum of n across buckets: {total_n}")
print(f"  Original entries: {entry_count}")
print(f"  Match: {total_n == entry_count}")
assert total_n == entry_count, "Mismatch!"

# ============================================================================
# Cross-check ground truth
# ============================================================================

print(f"\nSanity Check 2: Cross-check against file's precomputed ground truth")

recomputed_by_group = defaultdict(lambda: {'sum_opinion': 0.0, 'n': 0})
for (qid, cond, ed, gen, par, pk), b in finest.items():
    # Map dimension names to their values from the finest key tuple
    dim_values = {'education': ed, 'gender': gen, 'party': par, 'political_knowledge': pk}
    gid = "_".join(f"{d}_{dim_values[d]}" for d in DIMENSIONS)
    key = (gid, qid)
    recomputed_by_group[key]['sum_opinion'] += b['sum_opinion']
    recomputed_by_group[key]['n'] += b['n']

mismatches = 0
for key, recomp in recomputed_by_group.items():
    if key in gt_by_group:
        stored = gt_by_group[key]
        recomp_mean = recomp['sum_opinion'] / recomp['n']
        if abs(recomp_mean - stored['stored_mean']) > 1e-4 or recomp['n'] != stored['stored_n']:
            mismatches += 1
            if mismatches <= 3:
                print(f"  Mismatch: {key}")

if mismatches == 0:
    print(f"  ✓ All {len(recomputed_by_group)} groups match (within tolerance)")
else:
    print(f"  ✗ {mismatches} mismatches!")
    raise AssertionError("Cross-check failed!")

# ============================================================================
# PASS 2: Aggregate by dimension subsets
# ============================================================================

print(f"\nPass 2: Aggregating by {len(subsets)} dimension subsets...")

all_rows = []

for subset_idx, dim_subset in enumerate(subsets):
    agg = {}
    for (qid, cond, ed, gen, par, pk), b in finest.items():
        full_dims = {
            "education": ed,
            "gender": gen,
            "party": par,
            "political_knowledge": pk,
        }
        proj = tuple(full_dims[d] for d in dim_subset)
        agg_key = (qid, cond, proj)

        a = agg.get(agg_key)
        if a is None:
            a = {
                "sum_vec": np.zeros(REP_DIM, dtype=np.float64),
                "sum_opinion": 0.0,
                "n": 0,
                "question_label": b["question_label"],
            }
            agg[agg_key] = a

        a["sum_vec"] += b["sum_vec"]
        a["sum_opinion"] += b["sum_opinion"]
        a["n"] += b["n"]

    n_before = len(agg)
    for (qid, cond, proj), a in agg.items():
        if a["n"] < MIN_N:
            continue

        mean_vec = (a["sum_vec"] / a["n"]).astype(np.float32)
        mean_opinion = a["sum_opinion"] / a["n"]

        all_rows.append({
            "X": mean_vec,
            "y": mean_opinion,
            "n": a["n"],
            "n_dims": len(dim_subset),
            "dimension_subset": list(dim_subset),
            "dimension_values": dict(zip(dim_subset, proj)),
            "condition": cond,
            "question_id": qid,
            "question_label": a["question_label"],
        })

    n_after = sum(1 for (qid, cond, proj), a in agg.items() if a["n"] >= MIN_N)
    print(f"  Subset {subset_idx+1:2d}/{len(subsets)}: {len(dim_subset)}-D, "
          f"{n_before} groups, {n_after} after MIN_N={MIN_N}")

print(f"\nTotal rows after filtering: {len(all_rows)}")

if len(all_rows) == 0:
    raise ValueError("No rows passed filter!")

# ============================================================================
# Prepare output arrays
# ============================================================================

print(f"\nPreparing output arrays...")

X = np.stack([r["X"] for r in all_rows]).astype(np.float32)
y = np.array([r["y"] for r in all_rows], dtype=np.float32)
n_dims_arr = np.array([r["n_dims"] for r in all_rows], dtype=np.int8)
question_id_arr = np.array([r["question_id"] for r in all_rows], dtype=np.int16)

conditions_sorted = sorted({r["condition"] for r in all_rows})
cond_to_code = {c: i for i, c in enumerate(conditions_sorted)}
condition_code_arr = np.array([cond_to_code[r["condition"]] for r in all_rows], dtype=np.int8)

print(f"  X: {X.shape}, y: {y.shape}")
print(f"  Conditions: {conditions_sorted}")

# ============================================================================
# Save outputs
# ============================================================================

print(f"\nSaving {OUTPUT_NPZ}...")
np.savez_compressed(OUTPUT_NPZ, X=X, y=y, n_dims=n_dims_arr, question_id=question_id_arr, condition_code=condition_code_arr)

import os
print(f"  Size: {os.path.getsize(OUTPUT_NPZ) / 1024 / 1024:.1f} MB")

print(f"\nSaving {OUTPUT_METADATA}...")
meta_out = {
    "condition_code_map": cond_to_code,
    "n_rows": len(all_rows),
    "min_n": MIN_N,
    "dimensions": DIMENSIONS,
    "n_dims_distribution": {str(nd): int(sum(1 for r in all_rows if r["n_dims"] == nd)) for nd in range(1, 5)},
    "rows": [{
        "row_index": i,
        "n": r["n"],
        "n_dims": r["n_dims"],
        "dimension_subset": r["dimension_subset"],
        "dimension_values": r["dimension_values"],
        "condition": r["condition"],
        "question_id": r["question_id"],
        "question_label": r["question_label"],
        "y_mean": float(r["y"]),
    } for i, r in enumerate(all_rows)],
}
with open(OUTPUT_METADATA, 'w') as f:
    json.dump(meta_out, f, indent=2)

# ============================================================================
# Summary
# ============================================================================

print(f"\n" + "="*80)
print("Summary Statistics")
print("="*80)

print(f"\nRows by n_dims:")
for nd in range(1, 5):
    c = sum(1 for r in all_rows if r["n_dims"] == nd)
    print(f"  {nd}D: {c} ({100*c/len(all_rows):.1f}%)")

print(f"\nRows by condition:")
for cond in conditions_sorted:
    c = sum(1 for r in all_rows if r["condition"] == cond)
    print(f"  {cond:25s}: {c} ({100*c/len(all_rows):.1f}%)")

print(f"\nGround truth (y): [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}, std={y.std():.3f}")

ns = np.array([r["n"] for r in all_rows])
print(f"Group sizes: [{ns.min()}, {ns.max()}], mean={ns.mean():.1f}, std={ns.std():.1f}")

print(f"\n✅ Done! {len(all_rows)} rows ready for training.")
