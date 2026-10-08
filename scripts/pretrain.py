#!/usr/bin/env python3
"""Launch one of the three archived first-four-layer pretraining configurations."""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segment", type=int, choices=(512, 1024, 2048))
    parser.add_argument("--base-model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    matches = list((ROOT / "configs").glob(f"seg{args.segment}_pretrain/training_config.json"))
    if len(matches) != 1:
        parser.error(f"expected one segment {args.segment} config, found {len(matches)}")
    raw = json.loads(matches[0].read_text())
    out = args.output_dir or ROOT / "outputs" / matches[0].parent.name
    raw.update(base_model_name=args.base_model, tokenizer_name=args.base_model,
               output_dir=str(out), resume_from_checkpoint=(
                   str(out / "latest_checkpoint.pt") if args.resume else None))
    if args.resume and not (out / "latest_checkpoint.pt").is_file():
        parser.error(f"missing resume checkpoint in {out}")
    if args.dry_run:
        print(json.dumps({k: raw[k] for k in (
            "base_model_name", "segment_len", "neural_memory_layers",
            "neural_memory_segment_len", "neural_memory_batch_size",
            "batch_size", "micro_batch_size", "neural_mem_learning_rate",
            "total_tokens", "output_dir", "resume_from_checkpoint")}, indent=2))
        return
    from train_titan_llama import TrainingConfig, main as train
    field_names = {field.name for field in dataclasses.fields(TrainingConfig)}
    unknown = set(raw) - field_names
    if unknown:
        parser.error(f"config contains unknown TrainingConfig fields: {sorted(unknown)}")
    config = TrainingConfig(**raw)
    os.makedirs(out, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        (out / "training_config.json").write_text(
            json.dumps(dataclasses.asdict(config), indent=2) + "\n"
        )
    train(config)


if __name__ == "__main__":
    main()
