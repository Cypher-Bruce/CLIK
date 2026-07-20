"""Generation-based evaluation for Context2LoRA.

Supports two modes:

1. **Library mode** — imported by ``train.py`` during training.  Runs under
   FSDP with ``torch.distributed`` and uses the fixed-step generation loop
   so every rank issues the same number of collective operations.

2. **Standalone mode** — run from the command line after training::

       python evaluate.py \
           --checkpoint /path/to/output_dir/checkpoints/step_N \
           --eval_samples 500

   Loads the checkpoint (parameter generator or LoRA), evaluates on the
   training dataset and optionally a validation dataset, and prints metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
from collections import Counter
from dataclasses import dataclass

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging import get_logger
from src.utils.prompts import format_prompt, load_prompt_templates
from src.utils.torch_utils import PRECISION_TO_TYPE


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def token_level_metrics(
    prediction: str, ground_truth: str
) -> tuple[float, float, float]:
    """Return (precision, recall, F1) at the token level (SQuAD-style)."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ──────────────────────────────────────────────────────────────────────
# Generation helpers
# ──────────────────────────────────────────────────────────────────────

EVAL_MAX_NEW_TOKENS = 256


def aggregate_hidden_states(
    hidden_states, strategy: str, layer_index: int
) -> torch.Tensor:
    """Aggregate per-layer hidden states for the PG input.

    - ``single`` picks ``hidden_states[layer_index]`` (current behavior).
    - ``mean`` averages all (num_hidden_layers + 1) states, returning (B, T, D).
    - ``elmo`` stacks the states into (B, L+1, T, D); the PG performs the
      learnable softmax-weighted sum internally.
    """
    if strategy == "single":
        return hidden_states[layer_index]
    stacked = torch.stack(list(hidden_states), dim=1)  # (B, L+1, T, D)
    if strategy == "mean":
        return stacked.mean(dim=1)
    if strategy == "elmo":
        return stacked
    raise ValueError(f"Unknown pg_hidden_strategy: {strategy}")


def greedy_generate_fixed_steps(model, input_ids, attention_mask, n_steps, eos_ids):
    """Greedy autoregressive generation with a *fixed* number of forward calls.

    Always executes exactly ``n_steps`` model forward passes regardless of
    when EOS is produced.  This is essential for FSDP, where all ranks must
    issue the same sequence of collective operations.  Tokens generated
    after the first EOS are discarded.
    """
    generated = []
    past = None
    finished = False

    for _ in range(n_steps):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        next_id = out.logits[:, -1:, :].argmax(dim=-1)

        if not finished:
            tok = next_id.item()
            if tok in eos_ids:
                finished = True
            else:
                generated.append(tok)

        input_ids = next_id
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((1, 1))], dim=1
        )
        past = out.past_key_values

    return generated


def greedy_generate_early_stop(
    model, input_ids, attention_mask, max_new_tokens, eos_ids
):
    """Greedy autoregressive generation that stops at EOS.

    Used in standalone (non-distributed) mode where we don't need to keep
    all ranks in lock-step.
    """
    generated = []
    past = None

    for _ in range(max_new_tokens):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        next_id = out.logits[:, -1:, :].argmax(dim=-1)
        tok = next_id.item()
        if tok in eos_ids:
            break
        generated.append(tok)

        input_ids = next_id
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((1, 1))], dim=1
        )
        past = out.past_key_values

    return generated


# ──────────────────────────────────────────────────────────────────────
# Eval config — replaces the loose args bag
# ──────────────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    """Minimal configuration needed by the evaluation loop."""

    use_pg: bool = False
    pg_hidden_layer: int = -1
    pg_hidden_strategy: str = "single"
    max_seq_len: int = 8192
    eval_samples: int = 100
    seed: int = 42
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS


# ──────────────────────────────────────────────────────────────────────
# Core evaluation — works in both distributed and single-GPU modes
# ──────────────────────────────────────────────────────────────────────


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _evaluate_split(
    raw_dataset,
    format_fn,
    split_name: str,
    model,
    param_gen,
    tokenizer,
    cfg: EvalConfig,
    device: torch.device,
    torch_dtype: torch.dtype,
    prompt_templates: list[str] | None,
    output_dir: str | None = None,
):
    """Run generation-based evaluation on a single dataset split.

    Returns a dict of ``{split_name}/precision``, ``{split_name}/recall``,
    ``{split_name}/f1``.

    When *output_dir* is set, writes per-example predictions and metrics
    to ``{output_dir}/{split_name}_predictions.jsonl``.
    """
    distributed = _is_distributed()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    log = get_logger()

    indices = list(range(len(raw_dataset)))
    rng = random.Random(cfg.seed)
    rng.shuffle(indices)
    indices = indices[: cfg.eval_samples]

    # Pad so every rank processes the same number of samples
    remainder = len(indices) % world_size
    if remainder:
        indices += indices[: world_size - remainder]
    per_rank = len(indices) // world_size
    rank_start = rank * per_rank
    rank_indices = indices[rank_start : rank_start + per_rank]
    real_count = min(per_rank, max(0, cfg.eval_samples - rank_start))

    generate_fn = (
        greedy_generate_fixed_steps if distributed else greedy_generate_early_stop
    )

    eos_ids = getattr(model, "generation_config", None)
    eos_ids = getattr(eos_ids, "eos_token_id", None) if eos_ids else None
    if isinstance(eos_ids, int):
        eos_ids = {eos_ids}
    elif isinstance(eos_ids, list):
        eos_ids = set(eos_ids)
    else:
        eos_ids = {tokenizer.eos_token_id}

    sum_p, sum_r, sum_f1 = 0.0, 0.0, 0.0
    per_example_results: list[dict] = []
    total_samples = min(cfg.eval_samples, len(raw_dataset))
    eval_start = time.perf_counter()
    log_interval = max(1, min(10, len(rank_indices) // 10))

    for i, idx in enumerate(rank_indices):
        if i > 0 and i % log_interval == 0:
            elapsed = time.perf_counter() - eval_start
            per_sample = elapsed / i
            remaining = per_sample * (len(rank_indices) - i)
            global_done = min(i * world_size, total_samples)
            log.info(
                f"  [{split_name}] {global_done}/{total_samples} samples "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

        formatted = format_fn(raw_dataset[idx])
        context = formatted["context"]
        question = formatted["prompt"]
        ground_truth = formatted["completion"]

        if cfg.use_pg:
            ctx_enc = tokenizer(
                context,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=cfg.max_seq_len,
            ).to(device)
            ctx_out = model(
                input_ids=ctx_enc["input_ids"],
                attention_mask=ctx_enc["attention_mask"],
                output_hidden_states=True,
            )
            last_hidden = aggregate_hidden_states(
                ctx_out.hidden_states,
                cfg.pg_hidden_strategy,
                cfg.pg_hidden_layer,
            )
            del ctx_out

            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                pg_out = param_gen(
                    last_hidden, attention_mask=ctx_enc["attention_mask"]
                )
            pg_sd = {k: v for k, v in pg_out.items() if k != "param_norm"}
            model.set_pg_state_dict(pg_sd)

        # Build user_content to match the training data pipeline:
        # PG mode: question only (context is encoded separately above)
        # Non-PG mode: context + question in the user message
        if prompt_templates:
            if cfg.use_pg:
                user_content = format_prompt(prompt_templates, question)
            else:
                user_content = format_prompt(
                    prompt_templates, question, context=context or None
                )
        elif cfg.use_pg:
            user_content = question
        else:
            user_content = f"Context: {context}\n\nQuestion: {question}"

        messages = [{"role": "user", "content": user_content}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_enc = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).to(device)

        with torch.autocast(device_type="cuda", dtype=torch_dtype):
            generated = generate_fn(
                model,
                prompt_enc["input_ids"],
                prompt_enc["attention_mask"],
                cfg.max_new_tokens,
                eos_ids,
            )

        if cfg.use_pg:
            model.clear_pg_state_dict()

        answer = tokenizer.decode(generated, skip_special_tokens=True)
        p, r, f1 = token_level_metrics(answer, ground_truth)

        if i < real_count:
            sum_p += p
            sum_r += r
            sum_f1 += f1
            per_example_results.append(
                {
                    "idx": idx,
                    "context": context,
                    "prompt": question,
                    "ground_truth": ground_truth,
                    "prediction": answer,
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                }
            )

    # ---- Write per-example predictions to JSONL ----
    if output_dir is not None:
        if distributed:
            gathered: list[list[dict]] | None = (
                [None] * world_size if rank == 0 else None  # type: ignore[list-item]
            )
            dist.gather_object(per_example_results, gathered, dst=0)
            if rank == 0:
                all_results = [item for sublist in gathered for item in sublist]  # type: ignore[union-attr]
        else:
            all_results = per_example_results

        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{split_name}_predictions.jsonl")
            with open(out_path, "w") as f:
                for entry in all_results:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.info(
                f"  [{split_name}] Saved {len(all_results)} predictions to {out_path}"
            )

    stats = torch.tensor([sum_p, sum_r, sum_f1, float(real_count)], device=device)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total = max(stats[3].item(), 1.0)
    return {
        f"{split_name}/precision": stats[0].item() / total,
        f"{split_name}/recall": stats[1].item() / total,
        f"{split_name}/f1": stats[2].item() / total,
    }


@torch.no_grad()
def evaluate(
    model,
    param_gen,
    tokenizer,
    cfg: EvalConfig,
    device: torch.device,
    torch_dtype: torch.dtype,
    prompt_templates: list[str] | None = None,
    train_raw_dataset=None,
    train_format_fn=None,
    val_raw_dataset=None,
    val_format_fn=None,
    output_dir: str | None = None,
):
    """Generate answers on sampled train/val examples and compute metrics.

    Works in both distributed (FSDP) and single-GPU modes.  When
    ``torch.distributed`` is initialized, samples are sharded across ranks
    and metrics are all-reduced.  Otherwise runs everything on one device.

    When *output_dir* is set, per-example predictions and metrics are saved
    as JSONL files (one per split) inside that directory.
    """
    was_training = model.training
    pg_was_training = param_gen.training if param_gen is not None else False

    model.eval()
    if param_gen is not None:
        param_gen.eval()

    report = {}

    if train_raw_dataset is not None and train_format_fn is not None:
        report.update(
            _evaluate_split(
                train_raw_dataset,
                train_format_fn,
                "eval_train",
                model,
                param_gen,
                tokenizer,
                cfg,
                device,
                torch_dtype,
                prompt_templates,
                output_dir=output_dir,
            )
        )

    if val_raw_dataset is not None and val_format_fn is not None:
        report.update(
            _evaluate_split(
                val_raw_dataset,
                val_format_fn,
                "eval_val",
                model,
                param_gen,
                tokenizer,
                cfg,
                device,
                torch_dtype,
                prompt_templates,
                output_dir=output_dir,
            )
        )

    model.train(was_training)
    if param_gen is not None:
        param_gen.train(pg_was_training)

    return report


# ──────────────────────────────────────────────────────────────────────
# Standalone CLI
# ──────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="Post-training evaluation for Context2LoRA checkpoints"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint dir (e.g. .../checkpoints/step_500).",
    )
    p.add_argument(
        "--eval_samples",
        type=int,
        default=1280,
        help="Number of examples to sample per split.",
    )
    p.add_argument(
        "--eval_dataset_path",
        type=str,
        default=None,
        help="Path to a separate JSONL file for validation evaluation.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
    )
    p.add_argument(
        "--prompt_config",
        type=str,
        default=None,
        help="Override the prompt config from train_args.json.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    from src.core.generator import ParameterGenerator
    from src.core.mixin import inject_pg
    from src.data import _get_builder_cls

    args = _parse_args()
    device = torch.device(args.device)
    ckpt_dir = args.checkpoint

    # ---- Resolve experiment directory (two levels up from checkpoint) ----
    output_dir = os.path.dirname(os.path.dirname(ckpt_dir))

    train_args_path = os.path.join(output_dir, "train_args.json")
    if not os.path.isfile(train_args_path):
        raise FileNotFoundError(
            f"Cannot find train_args.json in {output_dir}. "
            "Make sure --checkpoint points to <output_dir>/checkpoints/step_N."
        )
    with open(train_args_path) as f:
        train_args = json.load(f)

    model_name = train_args["model_name_or_path"]
    use_pg = train_args.get("use_pg", False)
    hidden_layer = train_args.get("pg_hidden_layer", -1)
    hidden_strategy = train_args.get("pg_hidden_strategy", "single")
    max_seq_len = train_args.get("max_seq_len", 8192)
    precision = train_args.get("precision", "bf16")
    dataset_name = train_args.get("dataset_name", "squad")
    dataset_path = train_args.get("dataset_path")
    eval_dataset_path = args.eval_dataset_path or train_args.get("eval_dataset_path")

    # ---- Prompt templates ----
    prompt_templates = None
    if args.prompt_config:
        prompt_config = args.prompt_config
    else:
        prompt_config = train_args.get("eval_prompt_config")
    if prompt_config and os.path.isfile(prompt_config):
        prompt_templates = load_prompt_templates(prompt_config)
        print(f"Loaded {len(prompt_templates)} prompt templates from {prompt_config}")

    # ---- Load base model ----
    print(f"Experiment dir:  {output_dir}")
    print(f"Checkpoint:      {ckpt_dir}")
    print(f"Model:           {model_name}")
    print(f"Dataset:         {dataset_name}")
    print(f"Precision:       {precision}")
    print(f"Mode:            {'parameter_generation' if use_pg else 'lora/finetune'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = PRECISION_TO_TYPE.get(precision, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, trust_remote_code=True
    )

    param_gen = None
    if use_pg:
        pg_config_path = train_args.get("pg_config", "")
        if not os.path.isfile(pg_config_path):
            pg_config_path = os.path.join(output_dir, "pg_config.json")
        with open(pg_config_path) as f:
            pg_config = json.load(f)

        PGModelClass = inject_pg(pg_config, type(model))
        model.__class__ = PGModelClass
        model.enable_pg()

        pg_weights_path = os.path.join(ckpt_dir, "paramgen", "weights.pt")
        print(f"PG weights:      {pg_weights_path}")

        param_gen = ParameterGenerator(pg_config=pg_config)
        param_gen.load_state_dict(
            torch.load(pg_weights_path, map_location="cpu", weights_only=True)
        )
        param_gen.to(device=device, dtype=torch_dtype)
        param_gen.eval()

    model.to(device)
    model.eval()

    # ---- Build eval datasets ----
    eval_builder = _get_builder_cls(dataset_name)(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        use_pg=use_pg,
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
        prompt_templates=prompt_templates,
    )
    eval_builder.download()

    train_raw = eval_builder.load_raw()
    train_format_fn = eval_builder.format_example
    val_raw = eval_builder.load_raw_eval()
    val_format_fn = eval_builder.format_example if val_raw is not None else None

    cfg = EvalConfig(
        use_pg=use_pg,
        pg_hidden_layer=hidden_layer,
        pg_hidden_strategy=hidden_strategy,
        max_seq_len=max_seq_len,
        eval_samples=args.eval_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )

    splits_desc = ["train"]
    if val_raw is not None:
        splits_desc.append("val")
    print(
        f"Evaluating {cfg.eval_samples} samples per split ({' + '.join(splits_desc)})"
    )
    print()

    # ---- Run evaluation ----
    report = evaluate(
        model,
        param_gen,
        tokenizer,
        cfg,
        device,
        torch_dtype,
        prompt_templates,
        train_raw_dataset=train_raw,
        train_format_fn=train_format_fn,
        val_raw_dataset=val_raw,
        val_format_fn=val_format_fn,
        output_dir=ckpt_dir,
    )

    # ---- Print results ----
    print("=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    for key, value in sorted(report.items()):
        print(f"  {key:30s} {value:.4f}")
    print("=" * 60)

    # ---- Save results to checkpoint dir ----
    results_path = os.path.join(ckpt_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
