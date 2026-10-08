"""Answer-token SFT from a selected Titan checkpoint."""
import os, sys, json, random, argparse, dataclasses

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from train_titan_llama import TrainingConfig, main
from train_datasets import _ddp_shard

OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class QualityMCSFTDataset(IterableDataset):
    """QuALITY train split as A/B/C/D multiple choice.

    Prompt format:

        {article}\\n\\nQuestion: {q}\\nChoices:\\nA. ...\\nB. ...\\nAnswer:

    The target is " X" for the gold letter. Everything before it is masked,
    so training loss covers the answer letter alone.
    """

    def __init__(self, tokenizer_name, max_length, quality_examples=0,
                 data_path=None, seed=271828, **ignored):
        self.max_length = max_length
        self.examples = int(quality_examples)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        path = data_path or os.path.join(REPO, 'data/QuALITY.v1.0.1.htmlstripped.train')
        rows = []
        for line in open(path):
            art = json.loads(line)
            for q in art.get('questions', []):
                if 'gold_label' in q and len(q.get('options', [])) >= 2:
                    rows.append((art['article'], q))
        random.Random(seed).shuffle(rows)
        self.rows = rows
        self.truncated = 0
        print(f'[dataset] QuALITY-train MC SFT: {len(rows)} questions available, '
              f'{self.examples} samples requested, max_length={max_length}, '
              f'loss on the answer letter only', flush=True)

    def _build(self, article, q):
        """Tokenize the full prompt before keeping the final tokens."""
        choices = "\n".join(f"{OPTION_LABELS[i]}. {o.strip()}"
                             for i, o in enumerate(q['options']))
        query = f"Question: {q['question']}\nChoices:\n{choices}\nAnswer:"
        prompt = f"{article}\n\n{query}" if article else query

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        ans_ids = self.tokenizer.encode(f" {OPTION_LABELS[int(q['gold_label']) - 1]}",
                                        add_special_tokens=False)
        if len(ans_ids) != 1:
            raise ValueError(f'answer label must be one token, got {ans_ids}')

        if len(prompt_ids) > self.max_length - 1:      # keep the tail
            prompt_ids = prompt_ids[-(self.max_length - 1):]
            self.truncated += 1

        input_ids = prompt_ids + ans_ids
        labels = [-100] * len(prompt_ids) + ans_ids
        attention_mask = [1] * len(input_ids)
        pad = self.max_length - len(input_ids)
        if pad:
            input_ids += [self.tokenizer.pad_token_id] * pad
            labels += [-100] * pad
            attention_mask += [0] * pad
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self):
        # IterableDatasets are replicated to every DDP rank, so each rank must
        # take a disjoint slice itself or both GPUs train on identical batches.
        rank, world = _ddp_shard()
        rows = self.rows[rank::world]
        quota = self.examples // world
        produced = 0
        while produced < quota:
            for article, q in rows:
                sample = self._build(article, q)
                if sample is None:
                    continue
                yield sample
                produced += 1
                if produced >= quota:
                    return

DATASETS = {'quality_mc': QualityMCSFTDataset, 'longalign_loongrl': None}


def build(args):
    config_path = os.path.join(os.path.dirname(os.path.abspath(args.init_checkpoint)),
                               'training_config.json')
    src = json.load(open(config_path))
    fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    cfg = {k: v for k, v in src.items() if k in fields}
    for key in ('neural_memory_layers', 'segmented_attention_layers'):
        if isinstance(cfg.get(key), list):
            cfg[key] = tuple(cfg[key])
    if args.dataset == 'longalign_loongrl':
        ds_kwargs = dict(dataset_name='longalign_loongrl_sft',
                         longalign_sft_max_examples=args.longalign,
                         dataset_max_examples=args.loongrl,
                         quality_sft_max_examples=0,
                         dataset_config_name='hotpotqa_qwen_0_2500')
        n_samples = args.longalign + args.loongrl
    else:
        ds_kwargs = dict(dataset_name='quality_mc_sft',
                         quality_sft_max_examples=args.examples,
                         longalign_sft_max_examples=0,
                         dataset_max_examples=0)
        n_samples = args.examples
    if args.steps > 0:
        n_samples = args.steps * args.batch_size
    resume = os.path.join(args.output_dir, 'latest_checkpoint.pt')
    cfg.update(**ds_kwargs,
               pretrained_from_checkpoint=args.init_checkpoint,
               resume_from_checkpoint=resume if os.path.exists(resume) else None,
               output_dir=args.output_dir,
               num_epochs=1, use_ddp=args.ddp,
               sequence_length=args.seqlen,
               micro_batch_size=args.micro_batch_size, batch_size=args.batch_size,
               total_tokens=n_samples * args.seqlen,
               learning_rate=1e-6, min_learning_rate=1e-7,
               neural_mem_learning_rate=args.nmlr,
               min_neural_mem_learning_rate=args.nmlr / 40,
               neural_mem_warmup_start_learning_rate=args.nmlr / 40,
               neural_mem_schedule=True, warmup_steps=args.warmup,
               grad_clip=10.0, save_interval=args.save_every, log_interval=1,
               wandb_project=args.wandb_project,
               wandb_run_name=args.wandb_run_name or os.path.basename(args.output_dir),
               wandb_group=None, wandb_job_type=None)
    return TrainingConfig(**cfg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=os.path.join(REPO, 'outputs/quality_sft'))
    ap.add_argument('--dataset', default='quality_mc', choices=sorted(DATASETS))
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--examples', type=int, default=0)
    ap.add_argument('--ddp', action='store_true')
    ap.add_argument('--longalign', type=int, default=5120)
    ap.add_argument('--loongrl', type=int, default=2496)
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--init_checkpoint', required=True)
    ap.add_argument('--seqlen', type=int, default=16384)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--micro_batch_size', type=int, default=1)
    ap.add_argument('--nmlr', type=float, default=2e-4)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--save_every', type=int, default=4)
    ap.add_argument('--wandb_project', default='anonymous-titan')
    ap.add_argument('--wandb_run_name', default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    if args.dataset == 'longalign_loongrl':
        args.examples = args.longalign + args.loongrl
    elif args.examples <= 0:
        path = os.path.join(REPO, 'data/QuALITY.v1.0.1.htmlstripped.train')
        n_questions = sum(sum('gold_label' in q for q in json.loads(line).get('questions', []))
                          for line in open(path))
        args.examples = int(round(args.epochs * n_questions))
    if args.examples <= 0:
        raise SystemExit('No training examples requested')
    if args.examples < args.batch_size:
        raise SystemExit('Requested examples must cover at least one batch')
    cfg = build(args)
    print(f'dataset={args.dataset} steps={cfg.total_steps} output={cfg.output_dir}')
    if args.dry_run:
        sys.exit(0)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    main(cfg)
