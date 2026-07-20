"""Single-sample inference with a trained Context2LoRA checkpoint.

Supports both parameter-generation (PG) and LoRA/finetune checkpoints.

Usage (PG mode):
    python infer.py \
        --checkpoint /path/to/output_dir/checkpoints/step_N \
        --context "The Eiffel Tower was built between 1887 and 1889." \
        --prompt "When was the Eiffel Tower built?"

Usage (LoRA mode):
    python infer.py \
        --checkpoint /path/to/output_dir/checkpoints/step_N \
        --context "The Eiffel Tower was built between 1887 and 1889." \
        --prompt "When was the Eiffel Tower built?"

The checkpoint path is expected to live under an experiment output_dir that
contains ``train_args.json``.  The script auto-discovers model_name_or_path,
mode (PG vs LoRA), and weights from there.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import aggregate_hidden_states
from src.utils.prompts import format_prompt, load_prompt_templates
from src.utils.torch_utils import PRECISION_TO_TYPE


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference with a Context2LoRA checkpoint (PG or LoRA)"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint dir (e.g. .../checkpoints/step_500).",
    )
    p.add_argument("--context", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument(
        "--prompt_config",
        type=str,
        default=None,
        help="Override the prompt config from train_args.json.",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
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
    precision = train_args.get("precision", "bf16")

    # ---- Prompt templates ----
    prompt_templates = None
    if args.prompt_config:
        prompt_config = args.prompt_config
    else:
        prompt_config = train_args.get("eval_prompt_config")
    if prompt_config and os.path.isfile(prompt_config):
        prompt_templates = load_prompt_templates(prompt_config)
        print(f"Loaded {len(prompt_templates)} prompt templates from {prompt_config}")

    print(f"Experiment dir:  {output_dir}")
    print(f"Checkpoint:      {ckpt_dir}")
    print(f"Model:           {model_name}")
    print(f"Precision:       {precision}")
    print(f"Mode:            {'parameter_generation' if use_pg else 'lora/finetune'}")

    # ---- Load base model ----
    torch_dtype = PRECISION_TO_TYPE.get(precision, torch.bfloat16)

    print("Loading base model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, trust_remote_code=True
    )

    if use_pg:
        from src.core.generator import ParameterGenerator
        from src.core.mixin import inject_pg

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
    else:
        adapter_path = os.path.join(ckpt_dir, "adapter", "adapter_model.bin")
        if os.path.isfile(adapter_path):
            from peft import PeftModel

            peft_dir = os.path.dirname(adapter_path)
            model = PeftModel.from_pretrained(model, peft_dir)
            print(f"LoRA adapter:    {peft_dir}")
        else:
            print("No adapter found; using base model weights from DCP checkpoint.")

    model.to(device)
    model.eval()

    if use_pg:
        # ---- PG path: encode context -> generate LoRA weights ----
        print("Generating LoRA from context ...")
        ctx_ids = tokenizer(
            args.context, return_tensors="pt", add_special_tokens=False
        ).to(device)
        with torch.no_grad():
            ctx_out = model(
                input_ids=ctx_ids["input_ids"],
                attention_mask=ctx_ids["attention_mask"],
                output_hidden_states=True,
            )
            last_hidden = aggregate_hidden_states(
                ctx_out.hidden_states, hidden_strategy, hidden_layer
            )
            pg_out = param_gen(last_hidden, attention_mask=ctx_ids["attention_mask"])

        pg_state_dict = {k: v for k, v in pg_out.items() if k != "param_norm"}
        model.set_pg_state_dict(pg_state_dict)

    # ---- Build user message matching the training format ----
    if prompt_templates:
        if use_pg:
            user_content = format_prompt(prompt_templates, args.prompt)
        else:
            user_content = format_prompt(
                prompt_templates, args.prompt, context=args.context or None
            )
    elif use_pg:
        user_content = args.prompt
    else:
        user_content = f"Context: {args.context}\n\nQuestion: {args.prompt}"

    messages = [{"role": "user", "content": user_content}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

    print("Generating ...")
    with torch.no_grad():
        output_ids = model.generate(
            **prompt_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    if use_pg:
        model.clear_pg_state_dict()

    generated = output_ids[0, prompt_ids["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    print(f"\nContext: {args.context}")
    print(f"\nPrompt:  {args.prompt}")
    print(f"\nAnswer:  {answer}")


if __name__ == "__main__":
    main()
