import argparse
import datetime
import json
import os
import subprocess
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
import wandb
from torch import Tensor
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.core.generator import ParameterGenerator
from src.core.mixin import inject_pg
from src.data import (
    _get_builder_cls,
    build_data_collator,
    build_sft_dataset,
    prepare_sft_dataset,
)
from evaluate import EvalConfig, aggregate_hidden_states, evaluate
from src.utils.prompts import load_prompt_templates
from src.utils.logging import setup_logger
from src.utils.torch_utils import (
    PRECISION_TO_TYPE,
    ResumableDistributedSampler,
    get_rng_state,
    set_rng_state,
    set_seed,
)


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────


def format_eta(seconds: float) -> str:
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0:
        return f"{d}d{h:02d}h{m:02d}m{s:02d}s"
    return f"{h}h{m:02d}m{s:02d}s"


@torch.no_grad()
def compute_lora_param_norm(model) -> float:
    """Compute the L2 norm of LoRA adapter parameters (PEFT mode)."""
    total = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" in name:
            local = p.data.to_local() if hasattr(p.data, "to_local") else p.data
            total += local.float().norm().item() ** 2
    return total**0.5


@torch.no_grad()
def log_per_param_stats(model, report: dict):
    """Log mean, abs_mean, abs_max, norm for each trainable parameter and its
    gradient.  Handles FSDP DTensors via ``to_local()`` and all-reduces across
    ranks so the reported values are global."""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        _reduce_tensor_stats(param.data, report, f"parameter/{name}")
        if param.grad is not None:
            _reduce_tensor_stats(param.grad, report, f"gradient/{name}")


def _reduce_tensor_stats(tensor: Tensor, report: dict, prefix: str):
    t = tensor.to_local().detach() if hasattr(tensor, "to_local") else tensor.detach()
    if t.numel() == 0:
        stats = torch.zeros(5, device=t.device)
    else:
        stats = torch.stack(
            [
                torch.tensor(t.numel(), device=t.device, dtype=torch.float),
                t.float().sum(),
                t.float().abs().sum(),
                t.float().abs().max(),
                (t.float() ** 2).sum(),
            ]
        )
    dist.all_reduce(stats[:3], op=dist.ReduceOp.SUM)
    dist.all_reduce(stats[3:4], op=dist.ReduceOp.MAX)
    dist.all_reduce(stats[4:], op=dist.ReduceOp.SUM)

    report[f"{prefix}/mean"] = (stats[1] / stats[0]).item()
    report[f"{prefix}/abs_mean"] = (stats[2] / stats[0]).item()
    report[f"{prefix}/abs_max"] = stats[3].item()
    report[f"{prefix}/norm"] = stats[4].sqrt().item()


def compute_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_completion_mask: Tensor,
    teacher_completion_mask: Tensor,
    temperature: float,
) -> Tensor:
    """KL divergence between student and teacher on completion-token positions.

    Teacher and student sequences may have different lengths (teacher includes
    context in the prompt).  We extract logits at completion-masked positions
    from each — since the completion tokens are identical, the extracted
    vectors are aligned token-for-token.

    Each sample in the batch is processed independently to avoid misalignment
    when per-sample completion token counts differ between student and teacher.
    """
    # Shift for next-token prediction: logit at i predicts token at i+1
    s_logits = student_logits[:, :-1, :]
    t_logits = teacher_logits[:, :-1, :]
    s_mask = student_completion_mask[:, 1:]
    t_mask = teacher_completion_mask[:, 1:]

    batch_size = s_logits.shape[0]
    total_kl = student_logits.new_tensor(0.0)
    total_tokens = 0

    for i in range(batch_size):
        s_comp = s_logits[i][s_mask[i].bool()]  # (N_s, V)
        t_comp = t_logits[i][t_mask[i].bool()]  # (N_t, V)

        n = min(s_comp.shape[0], t_comp.shape[0])
        if n == 0:
            continue
        s_comp = s_comp[:n]
        t_comp = t_comp[:n]

        s_log_probs = F.log_softmax(s_comp / temperature, dim=-1)
        t_probs = F.softmax(t_comp / temperature, dim=-1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)
        total_kl = total_kl + kl.sum()
        total_tokens += n

    if total_tokens == 0:
        return student_logits.new_tensor(0.0)
    return (total_kl / total_tokens) * (temperature**2)


def maybe_apply_liger_kernel(model, args, logger):
    """Patch a Qwen3 model with Liger Kernel fused ops (RoPE, RMSNorm, SwiGLU).

    Step 1 of the kernel-acceleration plan. Only patches numerically-equivalent
    fused replacements that play nicely with our PG monkey-patch:

    * ``rope`` / ``rms_norm`` — drop-in fused kernels.
    * ``swiglu`` — replaces ``Qwen3MLP.forward`` with a fused SiLU·Mul.
      The submodules ``gate_proj`` / ``up_proj`` / ``down_proj`` are kept,
      and Liger's ``forward`` still calls them, so PG's per-projection LoRA
      additive injection (``convert_to_pg``) sees the LoRA term and the
      fused kernel sees the LoRA-augmented activations — semantically
      identical to the unpatched path.

    NOT enabled here:

    * ``cross_entropy`` / ``fused_linear_cross_entropy`` — would replace
      ``Qwen3ForCausalLM.forward`` so that ``out.logits`` becomes ``None``
      whenever ``labels`` are passed.  The distillation path
      (``compute_distillation_loss``) consumes ``student_out.logits``
      directly, so FLCE is handled separately in Step 2.
    """
    if not args.use_liger_kernel:
        return

    model_type = getattr(model.config, "model_type", None)
    if model_type != "qwen3":
        logger.warning(
            f"--use_liger_kernel was set but model_type={model_type!r} is not "
            "'qwen3'; skipping (only Qwen3 is wired up today)."
        )
        return

    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    apply_liger_kernel_to_qwen3(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        model=model,
    )

    sample_mlp = model.model.layers[0].mlp
    sample_norm = model.model.layers[0].input_layernorm
    final_norm = model.model.norm
    if sample_mlp._get_name() != LigerSwiGLUMLP.__name__:
        raise RuntimeError(
            f"Liger SwiGLU patch did not land: layer 0 mlp is {sample_mlp._get_name()}"
        )
    if sample_norm._get_name() != LigerRMSNorm.__name__:
        raise RuntimeError(
            f"Liger RMSNorm patch did not land on layer norm: {sample_norm._get_name()}"
        )
    if final_norm._get_name() != LigerRMSNorm.__name__:
        raise RuntimeError(
            f"Liger RMSNorm patch did not land on final norm: {final_norm._get_name()}"
        )
    logger.info(
        "Liger kernel applied to base model: rope+rms_norm+swiglu "
        f"(mlp={sample_mlp._get_name()}, norm={sample_norm._get_name()}, "
        f"layers patched={len(model.model.layers)})"
    )


def get_git_info():
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        )
        return branch, commit, dirty
    except Exception:
        return "unknown", "unknown", False


class _DistributedCheckpoint(Stateful):
    """Wraps a model and optimizer pair so that DCP can call
    ``state_dict`` / ``load_state_dict`` automatically."""

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


# ──────────────────────────────────────────────────────────────────────
# Checkpointing — LoRA path
# ──────────────────────────────────────────────────────────────────────


def save_checkpoint_lora(
    model, optimizer, scheduler, step, epoch, batch_offset, wandb_run_id, args, logger
):
    rank = dist.get_rank()
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{step}")

    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=ckpt_dir)

    torch.save(get_rng_state(), os.path.join(ckpt_dir, f"rng_rank_{rank}.pt"))

    adapter_sd = None
    if args.use_lora:
        # Gather LoRA params from FSDP shards onto rank 0.  Iterating
        # named_parameters() (rather than calling get_model_state_dict) keeps
        # us from materialising the full base-model state dict.
        from peft import get_peft_model_state_dict

        lora_full_sd = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                full = (
                    param.data.full_tensor()
                    if hasattr(param.data, "full_tensor")
                    else param.data
                )
                if rank == 0:
                    lora_full_sd[name] = full.cpu().clone()

        if rank == 0:
            # PEFT's helper renames keys to the layout that
            # PeftModel.from_pretrained expects (e.g. strips the
            # ".default." adapter-name suffix).  Without this step the
            # saved adapter_model.bin reloads silently as an all-zero
            # adapter -- see the now-fixed bug formerly tracked as
            # FINDINGS.md item 8.
            #
            # save_embedding_layers=False: we only train LoRA, never
            # the embedding.  PEFT's "auto" mode otherwise probes
            # model.state_dict() for the embedding tensor, which under
            # FSDP returns a sharded view and produces wrong weights.
            adapter_sd = get_peft_model_state_dict(
                model,
                state_dict=lora_full_sd,
                adapter_name="default",
                save_embedding_layers=False,
            )

    if rank == 0:
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        meta = {
            "step": step,
            "epoch": epoch,
            "batch_offset": batch_offset,
            "wandb_run_id": wandb_run_id,
        }
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        if args.use_lora and adapter_sd:
            adapter_dir = os.path.join(ckpt_dir, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            torch.save(adapter_sd, os.path.join(adapter_dir, "adapter_model.bin"))
            model.peft_config["default"].save_pretrained(adapter_dir)
            logger.info(f"LoRA adapter saved: {adapter_dir}")

    dist.barrier()

    if rank == 0:
        open(os.path.join(ckpt_dir, "COMPLETE"), "w").close()
        latest_file = os.path.join(args.output_dir, "checkpoints", "latest")
        with open(latest_file, "w") as f:
            f.write(f"step_{step}\n")

    dist.barrier()
    logger.info(f"Checkpoint saved: {ckpt_dir}")


def load_checkpoint_lora(model, optimizer, scheduler, ckpt_dir, logger):
    rank = dist.get_rank()

    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.load({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=ckpt_dir)
    set_state_dict(
        model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd
    )

    sched_path = os.path.join(ckpt_dir, "scheduler.pt")
    scheduler.load_state_dict(
        torch.load(sched_path, map_location="cpu", weights_only=True)
    )

    rng_path = os.path.join(ckpt_dir, f"rng_rank_{rank}.pt")
    set_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))

    with open(os.path.join(ckpt_dir, "metadata.json")) as f:
        meta = json.load(f)

    parent_run_id = meta.get("wandb_run_id")
    batch_offset = meta.get("batch_offset", 0)
    logger.info(
        f"Resumed from {ckpt_dir} (step={meta['step']}, epoch={meta['epoch']}"
        f", batch_offset={batch_offset}, parent_wandb_run={parent_run_id})"
    )
    return meta["step"], meta["epoch"], batch_offset, parent_run_id


# ──────────────────────────────────────────────────────────────────────
# Checkpointing — parameter generation path
# ──────────────────────────────────────────────────────────────────────


def save_checkpoint_pg(
    param_gen,
    optimizer,
    scheduler,
    step,
    epoch,
    batch_offset,
    wandb_run_id,
    args,
    logger,
):
    rank = dist.get_rank()
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    dcp.save(
        {"app": _DistributedCheckpoint(param_gen, optimizer)},
        checkpoint_id=ckpt_dir,
    )

    torch.save(get_rng_state(), os.path.join(ckpt_dir, f"rng_rank_{rank}.pt"))

    pg_dir = os.path.join(ckpt_dir, "paramgen")
    os.makedirs(pg_dir, exist_ok=True)
    sd = get_model_state_dict(
        param_gen, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    if rank == 0:
        torch.save(sd, os.path.join(pg_dir, "weights.pt"))
    del sd
    dist.barrier()

    if rank == 0:
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        meta = {
            "step": step,
            "epoch": epoch,
            "batch_offset": batch_offset,
            "wandb_run_id": wandb_run_id,
        }
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

    dist.barrier()

    if rank == 0:
        open(os.path.join(ckpt_dir, "COMPLETE"), "w").close()
        latest_file = os.path.join(args.output_dir, "checkpoints", "latest")
        with open(latest_file, "w") as f:
            f.write(f"step_{step}\n")

    dist.barrier()
    logger.info(f"Checkpoint saved: {ckpt_dir}")


def load_checkpoint_pg(param_gen, optimizer, scheduler, ckpt_dir, logger):
    rank = dist.get_rank()

    dcp.load(
        {"app": _DistributedCheckpoint(param_gen, optimizer)},
        checkpoint_id=ckpt_dir,
    )

    sched_path = os.path.join(ckpt_dir, "scheduler.pt")
    scheduler.load_state_dict(
        torch.load(sched_path, map_location="cpu", weights_only=True)
    )

    rng_path = os.path.join(ckpt_dir, f"rng_rank_{rank}.pt")
    set_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))

    with open(os.path.join(ckpt_dir, "metadata.json")) as f:
        meta = json.load(f)

    parent_run_id = meta.get("wandb_run_id")
    batch_offset = meta.get("batch_offset", 0)
    logger.info(
        f"Resumed from {ckpt_dir} (step={meta['step']}, epoch={meta['epoch']}"
        f", batch_offset={batch_offset}, parent_wandb_run={parent_run_id})"
    )
    return meta["step"], meta["epoch"], batch_offset, parent_run_id


def load_weights_only_lora(model, ckpt_dir, logger):
    """Initialize ``model`` from the model shard of a checkpoint and ignore
    everything else (optimizer, scheduler, RNG, step, epoch, wandb run id).

    Use this to warm-start a fresh training run from a previously trained
    model without inheriting its optimizer state or training progress.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    model_sd = get_model_state_dict(model)
    dcp.load({"model": model_sd}, checkpoint_id=ckpt_dir)
    set_model_state_dict(model, model_state_dict=model_sd)
    logger.info(f"Initialized model weights from {ckpt_dir} (weights only)")


def load_weights_only_pg(param_gen, ckpt_dir, logger):
    """Initialize ``param_gen`` from the model shard of a checkpoint and
    ignore everything else (optimizer, scheduler, RNG, step, epoch, wandb
    run id).

    Use this to warm-start a fresh training run from a previously trained
    parameter generator without inheriting its optimizer state or training
    progress.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    model_sd = get_model_state_dict(param_gen)
    dcp.load({"app": {"model": model_sd}}, checkpoint_id=ckpt_dir)
    set_model_state_dict(param_gen, model_state_dict=model_sd)
    logger.info(
        f"Initialized parameter generator weights from {ckpt_dir} (weights only)"
    )


def find_latest_checkpoint(output_dir: str):
    latest_file = os.path.join(output_dir, "checkpoints", "latest")
    if not os.path.isfile(latest_file):
        return None
    with open(latest_file) as f:
        tag = f.read().strip()
    ckpt_dir = os.path.join(output_dir, "checkpoints", tag)
    if not os.path.isfile(os.path.join(ckpt_dir, "COMPLETE")):
        return None
    return ckpt_dir


# ──────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Context2LoRA training with registered or JSONL datasets"
    )

    model = p.add_argument_group("Model")
    model.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B")
    model.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=list(PRECISION_TO_TYPE.keys()),
    )

    data = p.add_argument_group("Data")
    data.add_argument("--dataset_name", type=str, required=True)
    data.add_argument("--max_seq_len", type=int, required=True)
    data.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use only the first N examples from the dataset (default: use all).",
    )

    train = p.add_argument_group("Training")
    train.add_argument("--per_device_train_batch_size", type=int, default=2)
    train.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train.add_argument("--num_train_epochs", type=int, default=5)
    train.add_argument("--learning_rate", type=float, default=1e-5)
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--adam_beta1", type=float, default=0.9)
    train.add_argument("--adam_beta2", type=float, default=0.999)
    train.add_argument("--adam_epsilon", type=float, default=1e-8)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    train.add_argument("--warmup_steps", type=int, default=100)
    train.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train.add_argument("--gradient_checkpointing", action="store_true", default=False)
    train.add_argument(
        "--use_liger_kernel",
        action="store_true",
        default=False,
        help="Patch the base model with Liger Kernel fused ops "
        "(RoPE + RMSNorm + SwiGLU). Currently only Qwen3 is wired up. "
        "FLCE is intentionally OFF here — it conflicts with the distillation "
        "path that reads student_out.logits directly.",
    )
    train.add_argument(
        "--no_shard",
        action="store_true",
        default=False,
        help="Disable FSDP parameter sharding (NO_SHARD mode). "
        "Parameters stay replicated; only gradient all-reduce is used. "
        "Useful when the model fits in a single GPU's memory.",
    )

    lora = p.add_argument_group("LoRA / PEFT")
    lora.add_argument("--use_lora", action="store_true", default=False)
    lora.add_argument("--lora_r", type=int, default=16)
    lora.add_argument("--lora_alpha", type=int, default=32)
    lora.add_argument("--lora_dropout", type=float, default=0.05)
    lora.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA. If None, peft auto-detects linear layers.",
    )

    pg = p.add_argument_group("Parameter Generation")
    pg.add_argument("--use_pg", action="store_true", default=False)
    pg.add_argument(
        "--pg_config",
        type=str,
        default=None,
        help="Path to parameter generator config JSON.",
    )
    pg.add_argument(
        "--pg_hidden_layer",
        type=int,
        default=-1,
        help="Which layer's hidden states to feed to the parameter generator. "
        "Negative values index from the end (default: -1 = last layer). "
        "Only used when --pg_hidden_strategy=single.",
    )
    pg.add_argument(
        "--pg_hidden_strategy",
        type=str,
        default="single",
        choices=["single", "mean", "elmo"],
        help="How to aggregate base-model hidden states fed to the PG: "
        "'single' picks layer --pg_hidden_layer, 'mean' averages all "
        "(num_hidden_layers+1) states, 'elmo' learns a softmax-weighted "
        "sum with a scalar gamma (params live in the PG).",
    )
    pg.add_argument(
        "--pg_norm_penalty",
        type=float,
        default=0.0,
        help="L2 penalty weight on generated LoRA parameter norm (0 = disabled).",
    )

    distill = p.add_argument_group("Distillation")
    distill.add_argument(
        "--use_distillation",
        action="store_true",
        default=False,
        help="Use KL divergence distillation loss instead of NTP. Requires --use_pg.",
    )
    distill.add_argument(
        "--distill_temperature",
        type=float,
        default=2.0,
        help="Softmax temperature for KL divergence in distillation.",
    )

    data.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a JSONL file (required when --dataset_name is 'jsonl').",
    )
    data.add_argument(
        "--eval_dataset_path",
        type=str,
        default=None,
        help="Path to a separate JSONL file for validation evaluation. "
        "If not set, only the training set is evaluated.",
    )
    data.add_argument(
        "--prompt_config",
        type=str,
        default=None,
        help="Path to a JSON list of prompt templates with {input} placeholder. "
        "Context is prepended automatically for the teacher path. "
        "Must match the config used during data preparation.",
    )

    ckpt = p.add_argument_group("Checkpointing")
    ckpt.add_argument("--output_dir", type=str, required=True)
    ckpt.add_argument("--save_steps", type=int, default=500)
    ckpt.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Path to checkpoint dir, or "auto" to find the latest.',
    )
    ckpt.add_argument(
        "--init_weights_from",
        type=str,
        default=None,
        help="Path to a checkpoint dir whose model/param_gen weights should "
        "be loaded at the start of training. Optimizer, scheduler, RNG, "
        "step, epoch, and wandb run id are NOT restored. Ignored when "
        "--resume_from_checkpoint is also active and a checkpoint is found.",
    )

    eval_grp = p.add_argument_group("Evaluation")
    eval_grp.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="Run evaluation every N optimizer steps (0 = disabled).",
    )
    eval_grp.add_argument(
        "--eval_samples",
        type=int,
        default=100,
        help="Number of examples to sample per split for generation-based evaluation.",
    )
    eval_grp.add_argument(
        "--eval_prompt_config",
        type=str,
        default=None,
        help="Path to a JSON list of prompt templates for evaluation. "
        "If not set, no prompt template formatting is applied during eval.",
    )

    log = p.add_argument_group("Logging")
    log.add_argument("--logging_steps", type=int, default=10)
    log.add_argument(
        "--param_stats_steps",
        type=int,
        default=0,
        help="Log per-parameter stats every N steps (0 = disabled, expensive).",
    )
    log.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: do NOT create a W&B run. wandb.init runs in "
        "mode='disabled' so all wandb.log/define_metric/finish calls become "
        "no-ops; checkpoints record wandb_run_id=None so resuming this output "
        "dir without --debug starts a fresh run instead of forking from a "
        "fake one.",
    )

    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--dataloader_num_workers", type=int, default=4)

    args = p.parse_args()

    if args.use_pg and args.use_lora:
        p.error("--use_pg and --use_lora are mutually exclusive.")
    if args.use_pg and args.pg_config is None:
        p.error("--pg_config is required when --use_pg is set.")
    if args.use_distillation and not args.use_pg:
        p.error("--use_distillation requires --use_pg.")

    return args


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ---- Distributed init (env set by torchrun) ----
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=datetime.timedelta(minutes=30),
    )

    logger = setup_logger(args.output_dir)

    branch, commit, dirty = get_git_info()
    logger.info(f"Git: {branch}@{commit[:8]}{' (dirty)' if dirty else ''}")
    if dirty:
        logger.warning(
            "You have uncommitted changes in your git repository. "
            "This may cause issues with reproducibility. "
            "Please commit your changes before training."
        )
    logger.info(
        f"Distributed: world_size={world_size} rank={rank} local_rank={local_rank}"
    )
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")
    if rank == 0:
        args_path = os.path.join(args.output_dir, "train_args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # Seed for weight initialization — rank 0's parameters will be broadcast
    # to all ranks after construction, so the seed only matters on rank 0.
    # Per-rank seeds are set later, after FSDP sharding, for training randomness.
    set_seed(args.seed)

    # ---- Download model files on rank 0 first to avoid download races ----
    hf_cache = os.environ.get(
        "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
    )
    local_model_path = os.path.join(
        hf_cache, args.model_name_or_path.replace("/", "--")
    )
    if rank == 0:
        snapshot_download(args.model_name_or_path, local_dir=local_model_path)
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Prompt templates (for teacher tokenization in distillation) ----
    prompt_templates = None
    if args.prompt_config:
        prompt_templates = load_prompt_templates(args.prompt_config)
        logger.info(
            f"Loaded {len(prompt_templates)} prompt templates from {args.prompt_config}"
        )

    eval_prompt_templates = None
    if args.eval_prompt_config:
        eval_prompt_templates = load_prompt_templates(args.eval_prompt_config)
        logger.info(
            f"Loaded {len(eval_prompt_templates)} eval prompt templates "
            f"from {args.eval_prompt_config}"
        )

    # ---- Dataset & DataLoader ----
    # Rank 0 downloads remote datasets and converts JSONL files to Arrow
    # format (memory-mapped, instant load).  Other ranks wait at the barrier
    # and then load the cached Arrow tables directly.
    logger.info("Building dataset ...")
    if rank == 0:
        prepare_sft_dataset(
            args.dataset_name,
            dataset_path=args.dataset_path,
            eval_dataset_path=getattr(args, "eval_dataset_path", None),
        )
    dist.barrier()
    dataset = build_sft_dataset(
        args.dataset_name,
        tokenizer,
        args.max_seq_len,
        use_pg=args.use_pg,
        use_distillation=args.use_distillation,
        max_samples=args.max_samples,
        dataset_path=args.dataset_path,
        prompt_templates=prompt_templates,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    sampler = ResumableDistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=build_data_collator(tokenizer, use_pg=args.use_pg),
        num_workers=args.dataloader_num_workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Prepare raw datasets for generation-based evaluation ----
    eval_train_raw = None
    eval_train_format_fn = None
    eval_val_raw = None
    eval_val_format_fn = None
    eval_cfg = None

    if args.eval_steps > 0:
        eval_builder = _get_builder_cls(args.dataset_name)(
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            use_pg=args.use_pg,
            dataset_path=args.dataset_path,
            eval_dataset_path=getattr(args, "eval_dataset_path", None),
            prompt_templates=prompt_templates,
        )
        eval_train_raw = eval_builder.load_raw()
        eval_train_format_fn = eval_builder.format_example

        eval_val_raw_ds = eval_builder.load_raw_eval()
        if eval_val_raw_ds is not None:
            eval_val_raw = eval_val_raw_ds
            eval_val_format_fn = eval_builder.format_example

        eval_cfg = EvalConfig(
            use_pg=args.use_pg,
            pg_hidden_layer=args.pg_hidden_layer,
            pg_hidden_strategy=args.pg_hidden_strategy,
            max_seq_len=args.max_seq_len,
            eval_samples=args.eval_samples,
            seed=args.seed,
        )

        splits_desc = ["train"]
        if eval_val_raw is not None:
            splits_desc.append("val")
        logger.info(
            f"Eval enabled: every {args.eval_steps} steps, "
            f"{args.eval_samples} samples per split ({' + '.join(splits_desc)})"
        )

    # ---- Load parameter generation config (if applicable) ----
    pg_config = None
    if args.use_pg:
        with open(args.pg_config) as f:
            pg_config = json.load(f)
        logger.info(f"Parameter generation config: {json.dumps(pg_config, indent=2)}")

    # ---- Model ----
    logger.info(f"Loading model from: {local_model_path}")
    torch_dtype = PRECISION_TO_TYPE[args.precision]
    device = torch.device(f"cuda:{local_rank}")

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Patch Qwen3 with Liger fused kernels in-place. Must run before inject_pg
    # so that PG's per-projection LoRA wraps the same nn.Linear instances that
    # LigerSwiGLUMLP.forward() will call. Order also matters w.r.t. PEFT and
    # FSDP: Liger patches the *forward methods* on existing nn.Module instances,
    # so PEFT.get_peft_model and fully_shard see the already-patched modules.
    maybe_apply_liger_kernel(model, args, logger)

    if args.use_pg:
        # AutoModelForCausalLM resolves to the concrete class at runtime,
        # so we inject PG into the resolved class and swap in the subclass.
        PGModelClass = inject_pg(pg_config, type(model))
        model.__class__ = PGModelClass
        model.enable_pg()
        logger.info("Model loaded with parameter generation hooks enabled")

        param_gen = ParameterGenerator(pg_config=pg_config)
    else:
        param_gen = None

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if param_gen is not None:
            param_gen.enable_gradient_checkpointing()

    # ---- LoRA (static PEFT path) ----
    if args.use_lora:
        target_modules = args.lora_target_modules
        if target_modules and len(target_modules) == 1 and "," in target_modules[0]:
            target_modules = [m.strip() for m in target_modules[0].split(",")]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.to(device)

    # Broadcast randomly-initialized parameters from rank 0 so all ranks are
    # guaranteed identical before FSDP sharding, regardless of RNG differences.
    if args.use_lora:
        for name, param in model.named_parameters():
            if "lora_" in name:
                dist.broadcast(param.data, src=0)
        logger.info("Broadcast LoRA adapter parameters from rank 0")

    if args.use_pg:
        param_gen.to(device)
        for param in param_gen.parameters():
            dist.broadcast(param.data, src=0)
        logger.info("Broadcast ParameterGenerator parameters from rank 0")

    # ---- FSDP v2 (fully_shard) ----
    fsdp_mesh = None
    if args.no_shard:
        fsdp_mesh = init_device_mesh(
            "cuda", (world_size, 1), mesh_dim_names=("replicate", "shard")
        )
        logger.info("FSDP NO_SHARD: using 2D mesh (replicate=%d, shard=1)", world_size)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch_dtype,
        reduce_dtype=torch.float32,
    )
    base = model.base_model.model if args.use_lora else model
    for layer in base.model.layers:
        fully_shard(layer, mesh=fsdp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=fsdp_mesh, mp_policy=mp_policy)
    logger.info("Applied FSDP v2 to base model via fully_shard()")

    if args.use_pg:
        pg_mp_policy = MixedPrecisionPolicy(
            param_dtype=torch_dtype,
            reduce_dtype=torch.float32,
        )
        for block in param_gen.model.blocks:
            fully_shard(block, mesh=fsdp_mesh, mp_policy=pg_mp_policy)
        fully_shard(param_gen, mesh=fsdp_mesh, mp_policy=pg_mp_policy)
        logger.info("Applied FSDP v2 to parameter generator via fully_shard()")

    # Switch to per-rank seeds now that all weights are initialized identically.
    # This gives each rank independent dropout masks and data augmentation.
    set_seed(args.seed + rank)

    # ---- Freeze / trainability ----
    if args.use_pg:
        model.requires_grad_(False)
        param_gen.requires_grad_(True)

    # ---- Optimizer ----
    if args.use_pg:
        trainable_model = param_gen
    else:
        trainable_model = model

    decay_params, no_decay_params = [], []
    for name, param in trainable_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    num_trainable = sum(p.numel() for p in decay_params) + sum(
        p.numel() for p in no_decay_params
    )
    total_base_params = sum(p.numel() for p in model.parameters())
    if args.use_pg:
        total_pg_params = sum(p.numel() for p in param_gen.parameters())
        logger.info(f"Base model params: {total_base_params:,} (frozen)")
        logger.info(f"Parameter generator params: {total_pg_params:,} (trainable)")
        logger.info(
            f"Generated LoRA params per forward: {param_gen.get_lora_count():,}"
        )
    else:
        logger.info(
            f"Trainable params: {num_trainable:,} / {total_base_params:,} "
            f"({num_trainable / total_base_params * 100:.2f}%)"
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    # ---- LR Scheduler ----
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    max_train_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = args.warmup_steps

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # ---- Resume ----
    start_step = 0
    start_epoch = 0
    start_batch_offset = 0
    parent_wandb_id = None
    resume_path = args.resume_from_checkpoint
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(args.output_dir)

    if resume_path:
        if not os.path.isdir(resume_path):
            raise FileNotFoundError(f"Checkpoint directory not found: {resume_path}")
        if args.use_pg:
            start_step, start_epoch, start_batch_offset, parent_wandb_id = (
                load_checkpoint_pg(param_gen, optimizer, scheduler, resume_path, logger)
            )
        else:
            start_step, start_epoch, start_batch_offset, parent_wandb_id = (
                load_checkpoint_lora(model, optimizer, scheduler, resume_path, logger)
            )
    elif args.init_weights_from:
        if args.use_pg:
            load_weights_only_pg(param_gen, args.init_weights_from, logger)
        else:
            load_weights_only_lora(model, args.init_weights_from, logger)

    # ---- W&B ----
    if rank == 0:
        if args.debug:
            logger.warning(
                "--debug set: skipping W&B run creation (wandb.init mode='disabled')."
            )
            wandb.init(mode="disabled")
        else:
            wandb_kwargs = dict(
                project="Context2LoRA",
                name=f"{branch}_{commit[:8]}",
                config=vars(args),
                dir=args.output_dir,
                notes=f"Git: {branch}@{commit[:8]}{' (dirty)' if dirty else ''}",
                tags=[
                    f"dataset_{args.dataset_name}",
                    f"model_{args.model_name_or_path}",
                    f"branch_{branch}",
                    f"commit_{commit[:8]}",
                    "parameter_generation" if args.use_pg else "lora",
                ],
            )
            if parent_wandb_id:
                wandb_kwargs["fork_from"] = f"{parent_wandb_id}?_step={start_step}"
                wandb_kwargs["notes"] += (
                    f"\nForked from wandb run: {parent_wandb_id} at step {start_step}"
                )
                wandb_kwargs["tags"].append(
                    f"forked_from_{parent_wandb_id}@{start_step}"
                )
            wandb.init(**wandb_kwargs)

        wandb.define_metric("eval_train/precision", summary="max")
        wandb.define_metric("eval_train/recall", summary="max")
        wandb.define_metric("eval_train/f1", summary="max")
        wandb.define_metric("eval_val/precision", summary="max")
        wandb.define_metric("eval_val/recall", summary="max")
        wandb.define_metric("eval_val/f1", summary="max")

    # ---- Training summary ----
    logger.info("=" * 60)
    logger.info(
        f"  Mode                = {'parameter_generation' if args.use_pg else 'lora' if args.use_lora else 'full_finetune'}"
    )
    logger.info(
        f"  FSDP shard mode     = {'NO_SHARD' if args.no_shard else 'FULL_SHARD'}"
    )
    logger.info(f"  Num examples        = {len(dataset)}")
    logger.info(f"  Num epochs          = {args.num_train_epochs}")
    logger.info(f"  Device batch size   = {args.per_device_train_batch_size}")
    logger.info(f"  Grad accum steps    = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Effective batch     = {args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size}"
    )
    logger.info(f"  Steps per epoch     = {steps_per_epoch}")
    logger.info(f"  Total opt steps     = {max_train_steps}")
    logger.info(f"  Warmup steps        = {warmup_steps}")
    logger.info(f"  Resuming from step  = {start_step}")
    logger.info("=" * 60)

    # ---- Training loop ----
    global_step = start_step
    model.train()
    if args.use_pg:
        param_gen.train()

    for epoch in range(start_epoch, args.num_train_epochs):
        sampler.set_epoch(epoch)
        if epoch == start_epoch and start_batch_offset > 0:
            sampler.set_start_offset(start_batch_offset)
        else:
            sampler.set_start_offset(0)

        running_loss = 0.0
        micro_steps = 0
        timing = {
            "ctx": 0.0,
            "teacher": 0.0,
            "pg_fwd": 0.0,
            "model_fwd": 0.0,
            "bwd": 0.0,
        }
        step_start = time.perf_counter()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            if args.use_pg:
                # --- Parameter generation training step ---
                ctx_ids = batch.pop("context_input_ids").to(device)
                ctx_mask = batch.pop("context_attention_mask").to(device)

                if args.use_distillation:
                    teacher_ids = batch.pop("teacher_input_ids")
                    teacher_attn = batch.pop("teacher_attention_mask")
                    teacher_comp_mask = batch.pop("teacher_completion_mask")
                    student_comp_mask = batch.pop("completion_mask")
                else:
                    batch.pop("teacher_input_ids", None)
                    batch.pop("teacher_attention_mask", None)
                    batch.pop("teacher_completion_mask", None)
                    batch.pop("completion_mask", None)

                batch = {k: v.to(device) for k, v in batch.items()}

                # Step 1: extract context hidden states from frozen base model
                t0 = time.perf_counter()
                with torch.no_grad():
                    ctx_out = model(
                        input_ids=ctx_ids,
                        attention_mask=ctx_mask,
                        output_hidden_states=True,
                    )
                    last_hidden = aggregate_hidden_states(
                        ctx_out.hidden_states,
                        args.pg_hidden_strategy,
                        args.pg_hidden_layer,
                    )
                t1 = time.perf_counter()

                # Step 2: teacher forward (distillation only)
                t2 = t1
                if args.use_distillation:
                    teacher_ids = teacher_ids.to(device)
                    teacher_attn = teacher_attn.to(device)
                    teacher_comp_mask = teacher_comp_mask.to(device)
                    student_comp_mask = student_comp_mask.to(device)
                    with torch.no_grad():
                        teacher_out = model(
                            input_ids=teacher_ids,
                            attention_mask=teacher_attn,
                        )
                        teacher_logits = teacher_out.logits.detach()
                    del teacher_out
                    t2 = time.perf_counter()

                # Step 3: parameter generator forward
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    pg_output = param_gen(last_hidden, attention_mask=ctx_mask)

                pg_state_dict = {
                    k: v for k, v in pg_output.items() if k != "param_norm"
                }
                model.set_pg_state_dict(pg_state_dict)
                t3 = time.perf_counter()

                # Step 4: student forward with generated LoRA
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    student_out = model(**batch)

                if args.use_distillation:
                    loss = compute_distillation_loss(
                        student_out.logits,
                        teacher_logits,
                        student_comp_mask,
                        teacher_comp_mask,
                        args.distill_temperature,
                    )
                else:
                    loss = student_out.loss

                if args.pg_norm_penalty > 0:
                    loss = loss + args.pg_norm_penalty * pg_output["param_norm"]

                t4 = time.perf_counter()

                # Step 5: backward (gradients flow through PG via the LoRA outputs)
                (loss / args.gradient_accumulation_steps).backward()
                model.clear_pg_state_dict()
                t5 = time.perf_counter()

                timing["ctx"] += t1 - t0
                timing["teacher"] += t2 - t1
                timing["pg_fwd"] += t3 - t2
                timing["model_fwd"] += t4 - t3
                timing["bwd"] += t5 - t4

            else:
                # --- Standard SFT / LoRA training step ---
                batch = {k: v.to(device) for k, v in batch.items()}

                t0 = time.perf_counter()
                loss = model(**batch).loss
                t1 = time.perf_counter()
                (loss / args.gradient_accumulation_steps).backward()
                t2 = time.perf_counter()

                timing["model_fwd"] += t1 - t0
                timing["bwd"] += t2 - t1

            running_loss += loss.detach().item()
            micro_steps += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                trainable = param_gen if args.use_pg else model

                # Per-parameter stats (before clipping, expensive — separate interval)
                per_param_report = {}
                if (
                    args.param_stats_steps > 0
                    and global_step % args.param_stats_steps == 0
                ):
                    log_per_param_stats(trainable, per_param_report)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ---- Logging ----
                if global_step % args.logging_steps == 0:
                    step_time = (time.perf_counter() - step_start) / args.logging_steps
                    eta_str = format_eta((max_train_steps - global_step) * step_time)

                    avg_loss = running_loss / micro_steps
                    lr = scheduler.get_last_lr()[0]
                    gn = (
                        grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
                    )

                    if args.use_pg:
                        lora_norm = pg_output.get(
                            "param_norm", torch.tensor(0.0)
                        ).item()
                    else:
                        lora_norm = compute_lora_param_norm(model)

                    if args.use_pg:
                        avg_ctx = timing["ctx"] / micro_steps
                        avg_teacher = timing["teacher"] / micro_steps
                        avg_pg = timing["pg_fwd"] / micro_steps
                        avg_fwd = timing["model_fwd"] / micro_steps
                        avg_bwd = timing["bwd"] / micro_steps
                        logger.info(
                            f"[Epoch {epoch}] Step {global_step}/{max_train_steps} "
                            f"| Loss {avg_loss:.4f} | LR {lr:.2e} "
                            f"| GradNorm {gn:.4f} | LoRANorm {lora_norm:.2e} "
                            f"| Ctx {avg_ctx:.2f}s | Tch {avg_teacher:.2f}s "
                            f"| PG {avg_pg:.2f}s "
                            f"| Fwd {avg_fwd:.2f}s | Bwd {avg_bwd:.2f}s "
                            f"| Step {step_time:.2f}s | ETA {eta_str}"
                        )
                        if rank == 0:
                            report = {
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch,
                                "statistics/gradient_norm": gn,
                                "statistics/lora_weight_norm": lora_norm,
                                "performance/context_extraction_sec": avg_ctx,
                                "performance/teacher_forward_sec": avg_teacher,
                                "performance/pg_forward_sec": avg_pg,
                                "performance/model_forward_sec": avg_fwd,
                                "performance/backward_sec": avg_bwd,
                                "performance/step_time_sec": step_time,
                            }
                            if args.pg_norm_penalty > 0:
                                report["train/norm_penalty"] = (
                                    args.pg_norm_penalty * lora_norm
                                )
                            report.update(per_param_report)
                            wandb.log(report, step=global_step)
                    else:
                        avg_fwd = timing["model_fwd"] / micro_steps
                        avg_bwd = timing["bwd"] / micro_steps
                        logger.info(
                            f"[Epoch {epoch}] Step {global_step}/{max_train_steps} "
                            f"| Loss {avg_loss:.4f} | LR {lr:.2e} "
                            f"| GradNorm {gn:.4f} | LoRANorm {lora_norm:.2e} "
                            f"| Fwd {avg_fwd:.2f}s | Bwd {avg_bwd:.2f}s "
                            f"| Step {step_time:.2f}s | ETA {eta_str}"
                        )
                        if rank == 0:
                            report = {
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch,
                                "statistics/gradient_norm": gn,
                                "statistics/lora_weight_norm": lora_norm,
                                "performance/forward_time_sec": avg_fwd,
                                "performance/backward_time_sec": avg_bwd,
                                "performance/step_time_sec": step_time,
                            }
                            report.update(per_param_report)
                            wandb.log(report, step=global_step)

                    running_loss = 0.0
                    micro_steps = 0
                    timing = {k: 0.0 for k in timing}
                    step_start = time.perf_counter()

                # ---- Checkpointing ----
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    wandb_run_id = (
                        (None if args.debug else wandb.run.id) if rank == 0 else None
                    )
                    # batch_offset is in *per-rank index* units, matching
                    # ResumableDistributedSampler.set_start_offset which slices
                    # the per-rank index list directly.  Each micro-batch
                    # consumes per_device_train_batch_size indices.
                    batch_offset = (
                        sampler._start_offset
                        + (batch_idx + 1) * args.per_device_train_batch_size
                    )
                    if args.use_pg:
                        save_checkpoint_pg(
                            param_gen,
                            optimizer,
                            scheduler,
                            global_step,
                            epoch,
                            batch_offset,
                            wandb_run_id,
                            args,
                            logger,
                        )
                    else:
                        save_checkpoint_lora(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            epoch,
                            batch_offset,
                            wandb_run_id,
                            args,
                            logger,
                        )
                    ckpt_path = os.path.join(
                        args.output_dir, "checkpoints", f"step_{global_step}"
                    )
                    if rank == 0:
                        wandb.log(
                            {
                                "checkpoint/path": wandb.Html(
                                    f"<code>{ckpt_path}</code>"
                                )
                            },
                            step=global_step,
                        )

                # ---- Evaluation ----
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_output_dir = os.path.join(
                        args.output_dir, "eval_outputs", f"step_{global_step}"
                    )
                    eval_metrics = evaluate(
                        model,
                        param_gen,
                        tokenizer,
                        eval_cfg,
                        device,
                        torch_dtype,
                        eval_prompt_templates,
                        train_raw_dataset=eval_train_raw,
                        train_format_fn=eval_train_format_fn,
                        val_raw_dataset=eval_val_raw,
                        val_format_fn=eval_val_format_fn,
                        output_dir=eval_output_dir,
                    )
                    parts = " | ".join(f"{k} {v:.4f}" for k, v in eval_metrics.items())
                    logger.info(f"[Eval] Step {global_step} | {parts}")
                    if rank == 0:
                        wandb.log(eval_metrics, step=global_step)

        # Flush remainder micro-batches that didn't fill a full accumulation group
        remainder = (batch_idx + 1) % args.gradient_accumulation_steps
        if remainder != 0:
            trainable = param_gen if args.use_pg else model
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable.parameters(), args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        logger.info(f"Epoch {epoch} complete  (global_step={global_step})")

    # ---- Final checkpoint ----
    wandb_run_id = (None if args.debug else wandb.run.id) if rank == 0 else None
    if args.use_pg:
        save_checkpoint_pg(
            param_gen,
            optimizer,
            scheduler,
            global_step,
            epoch,
            0,
            wandb_run_id,
            args,
            logger,
        )
    else:
        save_checkpoint_lora(
            model,
            optimizer,
            scheduler,
            global_step,
            epoch,
            0,
            wandb_run_id,
            args,
            logger,
        )

    if rank == 0:
        wandb.finish()
    dist.barrier()
    dist.destroy_process_group()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
