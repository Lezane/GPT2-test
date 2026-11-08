#!/usr/bin/env python3
"""Six SGD experiments for Lezane/GPT2-test (current BPE-based main.py).

Place beside config.py, dataset.py, model.py, optimizer.py; run:
    python additional-sgd.py
    python additional-sgd.py --log-every 50

Defaults: 300 optimizer updates, LR = cfg.SGD_LEARNING_RATE * {0.5, 2},
seeds = {40, 41, 42}. Other training settings and FP32 match main.py.
Runs are sequential on cfg.DEVICE. Optional --precision bf16 uses CUDA BF16
mixed precision; use the same precision for every comparison.

L0 = most frequent 95% of observed non-special token TYPES; L1 = remainder.
This is NOT a 95% token-mass split or a 95% confidence interval. Membership
comes from dataset.build_frequency_groups, computed once for all six runs.
Metrics are token-weighted training-window metrics (before each update,
with dropout active), NOT held-out validation accuracy. Accuracy is top-1
next-token accuracy over the FULL vocabulary, reported as a fraction [0,1].
Special/unobserved types are excluded from L0/L1, as in main.py.

Each run has metrics.csv and status.json; summary.csv contains equal-seed
means, sample variances (ddof=1) and sample standard deviations at each
logging step, separately per LR. Missing groups have blank metrics, never
zero accuracy; n_seeds records the number contributing to each statistic.
final_summary.csv contains the FINAL WINDOW, not the entire-run average.
A timestamped output folder prevents overwriting earlier experiments.
"""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as cfg
from dataset import download_wikitext103, train_bpe_model, BPEWikiTextDataset, build_frequency_groups
from model import SimpleGPT
from optimizer import configure_sgd

SEEDS = (40, 41, 42)
LR_FACTORS = (0.5, 2.0)
MAJOR_FRACTION = 0.95
METRICS = tuple(f'{group}_{metric}' for group in ('total', 'L0', 'L1')
                for metric in ('loss', 'accuracy'))


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['lr_factor'], row['learning_rate'], row['step'])].append(row)
    output = []
    for (factor, lr, step), rows in sorted(groups.items()):
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError('Duplicate seed at the same LR and step')
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            variance = statistics.variance(values) if len(values) > 1 else None
            output.append(dict(lr_factor=factor, learning_rate=lr, step=step,
                               metric=metric, n_seeds=len(values),
                               expected_seeds=len(SEEDS), complete=len(values) == len(SEEDS),
                               mean=statistics.mean(values) if values else None,
                               sample_variance=variance,
                               sample_std=math.sqrt(variance) if variance is not None else None))
    return output


def refresh_summary(root, records, steps):
    rows = aggregate(records)
    write_csv(root / 'summary.csv', rows)
    write_csv(root / 'final_summary.csv', [row for row in rows if row['step'] == steps])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device, precision):
    return (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            if precision == 'bf16' else nullcontext())


@torch.no_grad()
def accumulate(totals, logits, labels, losses, is_major, is_minor):
    correct = logits.detach().argmax(dim=-1).eq(labels)
    masks = (torch.ones_like(labels, dtype=torch.bool), is_major[labels], is_minor[labels])
    for index, mask in enumerate(masks):
        totals[index, 0] += losses.detach()[mask].double().sum()
        totals[index, 1] += correct[mask].sum()
        totals[index, 2] += mask.sum()


def run_experiment(args, root, train_file, tokenizer, major, minor, seed, factor, records):
    lr = float(cfg.SGD_LEARNING_RATE) * factor
    run_dir = root / f'lr_x{factor:g}_seed_{seed}'
    run_dir.mkdir()
    info = dict(seed=seed, lr_factor=factor, learning_rate=lr, steps=args.steps)
    atomic_json(run_dir / 'status.json', dict(info, status='running'))
    seed_everything(seed)
    device = torch.device(cfg.DEVICE)
    vocab_size = tokenizer.get_vocab_size()
    loader = DataLoader(BPEWikiTextDataset(train_file, tokenizer, cfg.TARGET_LENGTH),
                        batch_size=cfg.SGD_TRAIN_BATCH_SIZE, num_workers=0)
    # Model creation and data ordering match main.py; reset RNG for every run.
    model = SimpleGPT(vocab_size=vocab_size, emb_dim=cfg.EMB_DIM,
                      num_heads=cfg.NUM_HEADS, depth=cfg.DEPTH).to(device)
    optimizer = configure_sgd(model, learning_rate=lr, momentum=cfg.SGD_MOMENTUM)
    major, minor = major.to(device), minor.to(device)
    model.train()
    iterator = iter(loader)
    totals = torch.zeros((3, 3), dtype=torch.float64, device=device)
    tokens_seen = 0
    window_start = 1
    started = time.monotonic()
    print(f'\nSGD lr={lr:g}, seed={seed}: {args.steps} optimizer updates', flush=True)
    with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = None
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.SGD_GRADIENT_ACC_STEPS):
                try:
                    inputs, labels = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    try:
                        inputs, labels = next(iterator)
                    except StopIteration as exc:
                        raise ValueError('Training file has no complete sequence') from exc
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.reshape(-1)
                with autocast_context(device, args.precision):
                    logits = model(inputs).reshape(-1, vocab_size)
                    losses = F.cross_entropy(logits, labels, reduction='none')
                    loss = losses.mean()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Non-finite loss at update {step}')
                (loss / cfg.SGD_GRADIENT_ACC_STEPS).backward()
                accumulate(totals, logits, labels, losses, major, minor)
                tokens_seen += labels.numel()
                del logits, losses, loss, inputs, labels
            optimizer.step()
            if step % args.log_every == 0 or step == args.steps:
                row = dict(seed=seed, lr_factor=factor, learning_rate=lr, step=step,
                           window_start_step=window_start, tokens_seen=tokens_seen,
                           elapsed_seconds=time.monotonic() - started)
                for group, (loss_sum, correct, count) in zip(('total', 'L0', 'L1'), totals.cpu().tolist()):
                    row[f'{group}_tokens'] = int(count)
                    row[f'{group}_loss'] = loss_sum / count if count else None
                    row[f'{group}_accuracy'] = correct / count if count else None
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
                records.append(row)
                refresh_summary(root, records, args.steps)
                print(f'  step {step}: L0 accuracy={row["L0_accuracy"]}, '
                      f'L1 accuracy={row["L1_accuracy"]}', flush=True)
                totals.zero_()
                window_start = step + 1
    if args.save_models:
        torch.save(dict(model_state_dict=model.state_dict(), **info), run_dir / 'final_model.pt')
    atomic_json(run_dir / 'status.json', dict(info, status='completed', tokens_seen=tokens_seen))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=300, help='Optimizer updates per run (default: 300)')
    parser.add_argument('--log-every', type=int, default=10, help='Updates per logging window (default: 10)')
    parser.add_argument('--output-dir', type=Path, default=Path('logs/additional_sgd'))
    parser.add_argument('--train-file', type=Path, help='Optional local training file; otherwise use repository download')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--save-models', action='store_true', help='Also save final model weights for each run')
    args = parser.parse_args()
    if min(args.steps, args.log_every, cfg.SGD_TRAIN_BATCH_SIZE, cfg.SGD_GRADIENT_ACC_STEPS) < 1:
        parser.error('Steps, log interval, batch size and accumulation must be positive')
    if not math.isfinite(cfg.SGD_LEARNING_RATE) or cfg.SGD_LEARNING_RATE <= 0:
        parser.error('SGD_LEARNING_RATE must be finite and positive')
    device = torch.device(cfg.DEVICE)
    if args.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        parser.error('--precision bf16 requires a CUDA device with BF16 support')
    root = args.output_dir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True, exist_ok=False)
    print(f'Results: {root.resolve()}', flush=True)
    seed_everything(SEEDS[0])
    train_file = str(args.train_file) if args.train_file else download_wikitext103()
    tokenizer = train_bpe_model(train_file)
    counts, major, minor = build_frequency_groups(train_file, tokenizer, MAJOR_FRACTION)
    np.savez(root / 'frequency_groups.npz', counts=counts.numpy(), is_major=major.numpy(), is_minor=minor.numpy())
    tokenizer.save(str(root / 'tokenizer.json'))
    settings = {name: str(getattr(cfg, name)) if isinstance(getattr(cfg, name), torch.device)
                else getattr(cfg, name) for name in dir(cfg) if name.isupper()}
    source_hashes = {}
    for name in ('config.py', 'dataset.py', 'model.py', 'optimizer.py', Path(__file__).name):
        path = Path(__file__).resolve().parent / name
        if path.exists():
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(root / 'manifest.json', dict(
        config=settings, seeds=SEEDS, lr_factors=LR_FACTORS,
        learning_rates=[cfg.SGD_LEARNING_RATE * factor for factor in LR_FACTORS],
        steps=args.steps, log_every=args.log_every, precision=args.precision,
        train_file=str(Path(train_file).resolve()), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        source_sha256=source_hashes, major_vocab_fraction=MAJOR_FRACTION,
        L0='most frequent 95% of observed non-special token types', L1='remaining 5%',
        metric_scope='token-weighted training windows; accuracy in [0,1]',
        variance='sample variance across seeds, ddof=1; separate for each LR and step',
        L0_types=int(major.sum()), L1_types=int(minor.sum()),
        L1_token_mass=float(counts[minor].sum() / counts.sum()),
        reproducibility='Seeds reset per run; CUDA bitwise determinism is not guaranteed'))
    records, failures = [], []
    for factor in LR_FACTORS:
        for seed in SEEDS:
            try:
                run_experiment(args, root, train_file, tokenizer, major, minor, seed, factor, records)
            except Exception as exc:
                failure = dict(seed=seed, lr_factor=factor, status='failed', error=str(exc))
                failures.append(failure)
                run_dir = root / f'lr_x{factor:g}_seed_{seed}'
                run_dir.mkdir(exist_ok=True)
                atomic_json(run_dir / 'status.json', failure)
                (run_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                print(f'FAILED: {failure}', flush=True)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    refresh_summary(root, records, args.steps)
    atomic_json(root / 'status.json', dict(status='failed' if failures else 'completed', failures=failures))
    for row in aggregate(records):
        if row['step'] == args.steps and row['metric'] in ('L0_accuracy', 'L1_accuracy'):
            print(f'LR {row["learning_rate"]:g} {row["metric"]}: '
                  f'mean={row["mean"]}, variance={row["sample_variance"]}, n={row["n_seeds"]}')
    print(f'All available results saved in {root.resolve()}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
