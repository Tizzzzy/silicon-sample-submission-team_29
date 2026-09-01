#!/usr/bin/env python3
"""
Extract all unique prompt representations ONCE, store in efficient binary format.

For each row index (1,584,000 total):
  1. Look up (profile_id, item_key) from prompt_index.json
  2. Assemble full prompt using assemble_prompt
  3. Batch-forward through Qwen 3.6 27B, extract last token from final layer
  4. Write directly to preallocated np.memmap (O(1) write, no dict-rewrite tax)
  5. Track completion via append-only completed_rows.jsonl (cheap resume)

Output:
  - representations.dat (np.memmap, shape=(1_584_000, 5120), dtype=float16)
    Each row: one (profile_id, item_key)'s last-token representation
  - completed_rows.jsonl (append-only log of completed row indices)

Resume-on-failure: if completed_rows.jsonl exists, skip those rows and continue.

Usage:
  python extract_unique_representations.py \\
    --prompt_index prompt_index.json \\
    --prompts_dir ../prompt \\
    --model_path /projects/p32143/cache/qwen36_27b \\
    --output_dir . \\
    --batch_size 16 \\
    --device cuda \\
    [--max_rows 2000]  # optional, for testing

Expected time estimates (with real throughput logged every N batches):
  - batch_size=4: ~20-30 sec per batch of 4 → ~132-200 minutes for full 1,584,000
  - batch_size=8: ~40-50 sec per batch of 8 → ~132-165 minutes for full 1,584,000
  Actual time varies by prompt length and GPU utilization.

Output storage:
  - representations.dat: 1,584,000 × 5,120 × 2 bytes (float16) = 15.3 GB
  - completed_rows.jsonl: ~30 MB (one integer per line, 1,584,000 lines)
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
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt.assemble import assemble_prompt, load_items_meta

log = logging.getLogger("extract_unique_representations")


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
        device_map="auto",
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
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for prompt in prompts
    ]

    # Tokenize batch
    inputs = processor.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        processor_kwargs={"padding": True, "return_tensors": "pt"},
    ).to(device)

    # messages_batch = [
    #     {"role": "user", "content": [{"type": "text", "text": prompt}]}
    #     for prompt in prompts
    # ]

    # # Tokenize batch
    # inputs = processor.apply_chat_template(
    #     messages_batch,
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_dict=True,
    #     processor_kwargs={"padding": True, "return_tensors": "pt"},
    # ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract last token representation from final layer for each item in batch
    last_layer_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

    attention_mask = inputs["attention_mask"]
    # Find the index of the last non-pad token for each sequence
    sequence_lengths = attention_mask.sum(dim=1) - 1
    representations = []

    for i in range(last_layer_states.shape[0]):
        last_token_rep = last_layer_states[i, sequence_lengths[i], :].cpu().float().numpy()
        representations.append(last_token_rep)

    # Cleanup GPU memory
    del inputs, outputs, last_layer_states, messages_batch
    if device == "cuda":
        torch.cuda.empty_cache()

    return representations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_index", default="prompt_index.json",
                       help="Path to prompt_index.json from build_unique_prompt_index.py")
    parser.add_argument("--prompts_dir", default="../prompt",
                       help="Path to prompts directory")
    parser.add_argument("--model_path", default="/projects/p32143/cache/qwen36_27b")
    parser.add_argument("--output_dir", default=".",
                       help="Output directory for representations.dat and completed_rows.jsonl")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for forward passes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_rows", type=int, default=None,
                       help="Limit extraction to first N rows (for testing)")
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 78)
    log.info("EXTRACT UNIQUE REPRESENTATIONS (1,584,000 PROMPTS, ONCE)")
    log.info("=" * 78)
    log.info(f"Prompt index: {args.prompt_index}")
    log.info(f"Output dir: {args.output_dir}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Device: {args.device}")
    if args.max_rows:
        log.info(f"Test mode: limiting to first {args.max_rows} rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt index
    log.info("Loading prompt index...")
    with open(args.prompt_index) as f:
        prompt_index = json.load(f)
    log.info(f"Loaded {len(prompt_index):,} indexed prompts")
    total_prompts = len(prompt_index)

    if args.max_rows:
        total_prompts = min(args.max_rows, total_prompts)
        log.info(f"Test mode: processing first {total_prompts:,} prompts")

    # Load items metadata and profile contexts
    log.info("Loading prompts metadata...")
    items_meta = load_items_meta(Path(args.prompts_dir) / "items_meta.json")
    prompts_dict = {}
    with open(Path(args.prompts_dir) / "prompts.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            prompts_dict[rec["profile_id"]] = rec
    log.info(f"Loaded {len(prompts_dict):,} profile contexts")

    # Load model and processor
    model, processor = load_model_and_processor(args.model_path, args.device)

    # Create or load completed rows tracking
    completed_rows_file = output_dir / "completed_rows.jsonl"
    completed_rows = set()
    if completed_rows_file.exists():
        log.info(f"Loading prior completions from {completed_rows_file}...")
        with open(completed_rows_file) as f:
            for line in f:
                completed_rows.add(int(line.strip()))
        log.info(f"Loaded {len(completed_rows):,} completed rows, will resume from there")

    # Create preallocated memmap for representations
    representations_file = output_dir / "representations.dat"
    log.info(f"Creating memmap: {representations_file} (shape={total_prompts,5120}, dtype=float16)")
    if representations_file.exists():
        log.info(f"Memmap file already exists, will append to it")
    representations_memmap = np.memmap(
        representations_file,
        dtype=np.float16,
        mode='w+' if not representations_file.exists() else 'r+',
        shape=(total_prompts, 5120)
    )

    # Sort index keys for consistent iteration
    sorted_index_keys = sorted(prompt_index.keys())
    rows_to_process = [i for i, key in enumerate(sorted_index_keys[:total_prompts])
                       if i not in completed_rows]

    rows_to_process = [r for r in rows_to_process if r < 316800]

    log.info(f"Need to process {len(rows_to_process):,} rows (skipping {len(completed_rows):,} completed)")

    # Extract representations
    log.info("Extracting representations...")
    prompts_per_second_history = []

    batch_start_time = time.time()

    for batch_idx, batch_start in enumerate(range(0, len(rows_to_process), args.batch_size)):
        batch_end = min(batch_start + args.batch_size, len(rows_to_process))
        batch_row_indices = rows_to_process[batch_start:batch_end]

        # Assemble prompts for this batch
        batch_prompts = []
        for row_idx in batch_row_indices:
            key = sorted_index_keys[row_idx]
            profile_id, item_key = key.split("__", 1)

            try:
                profile_record = prompts_dict[profile_id]
                context = profile_record["context"]
                item_meta = items_meta[item_key]
                full_prompt = assemble_prompt(context, item_meta)
                batch_prompts.append((row_idx, full_prompt))
            except Exception as e:
                log.warning(f"Failed to assemble row {row_idx} ({key}): {e}")
                continue

        # Extract representations for entire batch
        if batch_prompts:
            try:
                prompt_texts = [p[1] for p in batch_prompts]
                batch_reps = extract_batch_representations(model, processor, prompt_texts, args.device)

                # Write to memmap (O(1) per prompt)
                for (row_idx, _), rep in zip(batch_prompts, batch_reps):
                    representations_memmap[row_idx] = rep
                    representations_memmap.flush()  # Flush to disk after each row

                    # Log completion
                    with open(completed_rows_file, "a") as f:
                        f.write(f"{row_idx}\n")

                n_completed = len(batch_prompts)
                # n_completed_total = len(completed_rows) + n_completed
                n_completed_total = len(completed_rows) + batch_end
                elapsed = time.time() - batch_start_time
                batch_start_time = time.time()

                # Log throughput
                # throughput = len(batch_prompts) / ((batch_end - batch_start) * 0.1)  # rough est.
                throughput = len(batch_prompts) / max(elapsed, 0.001)
                prompts_per_second_history.append(throughput)
                avg_throughput = np.mean(prompts_per_second_history[-10:])  # moving average

                if (batch_idx + 1) % 10 == 0:
                    eta_secs = (total_prompts - n_completed_total) / max(avg_throughput, 0.1)
                    eta_mins = eta_secs / 60
                    log.info(f"Batch {batch_idx + 1}: {n_completed_total:,} / {total_prompts:,} "
                            f"({100*n_completed_total/total_prompts:.1f}%) "
                            f"ETA: {eta_mins:.0f} min ({avg_throughput:.1f} prompts/sec)")

                    # eta_secs = (len(rows_to_process) - n_completed_total) / max(avg_throughput, 0.1)
                    # eta_mins = eta_secs / 60
                    # log.info(f"Batch {batch_idx + 1}: {n_completed_total:,} / {len(rows_to_process):,} "
                    #         f"({100*n_completed_total/len(rows_to_process):.1f}%) "
                    #         f"ETA: {eta_mins:.0f} min ({avg_throughput:.1f} prompts/sec)")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.error(f"OOM on batch {batch_start}-{batch_end}")
                    log.error(f"Suggestion: Reduce batch_size (currently {args.batch_size})")
                    raise
                else:
                    log.warning(f"Failed to extract batch {batch_start}-{batch_end}: {e}")
                    continue

    # Final summary
    log.info("-" * 78)
    log.info("SUMMARY")
    log.info("-" * 78)
    log.info(f"Representations file: {representations_file}")
    log.info(f"Size: {representations_file.stat().st_size / (1024**3):.1f} GB")
    log.info(f"Shape: {representations_memmap.shape}")
    log.info(f"Dtype: {representations_memmap.dtype}")
    log.info(f"Completed rows: {len(completed_rows) + len(rows_to_process):,} / {total_prompts:,}")

    log.info("\n" + "=" * 78)
    log.info("✓ EXTRACTION COMPLETE")
    log.info("=" * 78)


if __name__ == "__main__":
    # setup_logging()
    main()
