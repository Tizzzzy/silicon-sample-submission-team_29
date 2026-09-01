#!/usr/bin/env python3
"""
OOM-SAFE: Extract group-level representations with RESUME FROM EXISTING JSON.

CRITICAL CHANGES from batched version:
  - Load existing group_representations_main.json if it exists (resume from there)
  - Use smaller batch sizes (4-8) to avoid OOM
  - Aggressive GPU memory cleanup between batches
  - SAME OUTPUT FILE as input (append/update only new groups)

For each cell (condition × outcome):
  1. If already in output JSON → skip it
  2. Otherwise, batch-load all (profile_id, item_key) pairs in the cell
  3. Process in SMALL sub-batches (batch_size=4-8):
     - Assemble all prompts in batch
     - Forward through Qwen once with all prompts
     - Extract representations for all prompts
     - Clear GPU cache immediately
  4. Average all representations for the group
  5. Save to output JSON immediately (append/overwrite)
  6. Move to next group

Usage:
  python extract_group_representations_main_oom_safe.py \\
    --cells_file ../prompt/cells_main_prompts_tier2.json \\
    --prompts_dir ../prompt \\
    --model_path /projects/p32143/cache/qwen36_27b \\
    --output_file group_representations_main.json \\
    --batch_size 8 \\
    --device cuda

RECOMMENDED BATCH SIZES (by your VRAM):
  - batch_size=2  → ~15GB/pass (safest)
  - batch_size=4  → ~25GB/pass (recommended)
  - batch_size=8  → ~45GB/pass (if stable)
  - batch_size=16 → ~75GB/pass (risky on A100)

Expected speedup: 2-3× vs. serial (conservative, stable)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt.assemble import assemble_prompt, load_items_meta

log = logging.getLogger("extract_group_representations_main_oom_safe")


def setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(console)


def load_model_and_processor(model_path: str, device: str):
    """Load Qwen 3.6 27B model and processor."""
    log.info(f"Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    log.info(f"Model loaded on {device}")
    return model, processor


def extract_batch_representations(
    model,
    processor,
    prompts: List[str],
    device: str
) -> List[np.ndarray]:
    """
    Extract representations for a batch of prompts in one forward pass.
    Includes aggressive memory cleanup.

    Args:
        model: Qwen model
        processor: Tokenizer/processor
        prompts: List of prompt strings
        device: Device to run on

    Returns:
        List of numpy arrays, each shape (5120,)
    """
    # Format all prompts as messages
    messages_batch = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
        for prompt in prompts
    ]

    # Tokenize batch (with proper processor_kwargs for batch padding)
    inputs = processor.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        processor_kwargs={"padding": True, "return_tensors": "pt"},
    ).to(device)

    # Forward pass for entire batch
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract last token representation from final layer for each item in batch
    last_layer_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
    representations = []

    for i in range(last_layer_states.shape[0]):
        last_token_rep = last_layer_states[i, -1, :].cpu().float().numpy()
        representations.append(last_token_rep)

    # CRITICAL: Cleanup GPU memory after extraction
    del inputs, outputs, last_layer_states, messages_batch
    if device == "cuda":
        torch.cuda.empty_cache()

    return representations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells_file", default="../prompt/cells_main_prompts_tier2.json")
    parser.add_argument("--prompts_dir", default="../prompt")
    parser.add_argument("--model_path", default="/projects/p32143/cache/qwen36_27b")
    parser.add_argument("--output_file", default="group_representations_main.json")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (4-8 safe, 16+ risky)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 78)
    log.info("EXTRACT GROUP-LEVEL REPRESENTATIONS (OOM-SAFE, RESUME FROM JSON)")
    log.info("=" * 78)
    log.info(f"Cells file: {args.cells_file}")
    log.info(f"Output file: {args.output_file}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Device: {args.device}")

    # Load model and processor
    model, processor = load_model_and_processor(args.model_path, args.device)

    # Load items metadata and prompts
    log.info("Loading prompts metadata...")
    items_meta = load_items_meta(Path(args.prompts_dir) / "items_meta.json")
    prompts_dict = {}
    with open(Path(args.prompts_dir) / "prompts.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            prompts_dict[rec["profile_id"]] = rec
    log.info(f"Loaded {len(prompts_dict)} profile contexts")

    # Load cells grouping
    log.info("Loading cells grouping...")
    with open(args.cells_file) as f:
        cells = json.load(f)
    log.info(f"Loaded {len(cells)} cells")

    # Load existing output JSON (resume from there)
    group_representations = {}
    if Path(args.output_file).exists():
        log.info(f"Loading existing results from {args.output_file}...")
        with open(args.output_file) as f:
            group_representations = json.load(f)
        log.info(f"Loaded {len(group_representations)} existing group representations")

    # Extract representations for each group
    log.info("Extracting group-level representations...")
    total_prompts_processed = 0
    cells_to_process = [k for k in sorted(cells.keys()) if k not in group_representations]
    log.info(f"Need to process {len(cells_to_process)} new cells")

    for cell_idx, cell_key in enumerate(cells_to_process):
        if (cell_idx + 1) % 25 == 0:
            log.info(f"Processing cell {cell_idx + 1}/{len(cells_to_process)} ({cell_key})...")

        prompts_in_cell = cells[cell_key]
        representations = []

        # Process prompts in sub-batches
        for batch_start in range(0, len(prompts_in_cell), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(prompts_in_cell))
            batch_prompt_infos = prompts_in_cell[batch_start:batch_end]

            # Assemble prompts for this batch
            batch_prompts = []
            for prompt_info in batch_prompt_infos:
                profile_id = prompt_info["profile_id"]
                item_key = prompt_info["item_key"]

                try:
                    profile_record = prompts_dict[profile_id]
                    context = profile_record["context"]
                    item_meta = items_meta[item_key]
                    full_prompt = assemble_prompt(context, item_meta)
                    batch_prompts.append(full_prompt)
                except Exception as e:
                    log.warning(f"Failed to assemble {profile_id}/{item_key}: {e}")
                    continue

            # Extract representations for entire batch at once
            if batch_prompts:
                try:
                    batch_reps = extract_batch_representations(model, processor, batch_prompts, args.device)
                    representations.extend(batch_reps)
                    total_prompts_processed += len(batch_reps)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log.error(f"OOM on batch {batch_start}-{batch_end} of {cell_key}")
                        log.error(f"Suggestion: Reduce batch_size (currently {args.batch_size})")
                        raise
                    else:
                        log.warning(f"Failed to extract batch {batch_start}-{batch_end}: {e}")
                        continue

        # Average representations for this group
        if representations:
            avg_representation = np.mean(representations, axis=0)
            group_representations[cell_key] = avg_representation.tolist()

            # Save immediately (append to existing JSON)
            with open(args.output_file, "w") as f:
                json.dump(group_representations, f, indent=2)
            print(f"Saved group representation for {cell_key} (total groups processed: {len(group_representations)})")
        else:
            log.warning(f"No representations extracted for {cell_key}")

        # Free memory
        del representations

    # Final status
    log.info("-" * 78)
    log.info("SUMMARY")
    log.info("-" * 78)
    log.info(f"Total cells: {len(cells)}")
    log.info(f"Cells with representations: {len(group_representations)}")
    log.info(f"Total prompts processed this run: {total_prompts_processed:,}")
    log.info(f"Output file: {args.output_file}")

    log.info("\n" + "=" * 78)
    log.info("✓ EXTRACTION COMPLETE")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
