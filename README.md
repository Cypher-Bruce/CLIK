# Context2LoRA

Context2LoRA turns a context passage into LoRA parameters for a frozen language
model. After the adapter is generated, the model can answer questions about the
context without receiving the passage again in its prompt.

> [!NOTE]
> This repository contains research code. Interfaces and checkpoint formats may
> change, and pretrained Context2LoRA weights are not included.

## How it works

```text
Context ──> frozen base model ──> hidden states ──> parameter generator ──> LoRA weights
                                                                              |
Question ─────────────────────────────> base model with generated LoRA ──> Answer
```

The parameter generator serializes the target LoRA matrices as two-dimensional
tokens, processes them together with context representations, and reconstructs
per-layer LoRA A and B matrices. The current configurations target the MLP
projections of Qwen 3 and Qwen 3.5 models.

Training supports two stages:

1. Parameter-generator pretraining with next-token prediction
   (`USE_DISTILLATION=0`).
2. Post-distillation, where a teacher sees the context and question while the
   student answers from the question and generated LoRA parameters. This stage
   minimizes KL divergence on completion tokens and can start from a pretrained
   parameter-generator checkpoint.

## Requirements

- Linux
- Python 3.12
- A CUDA-capable GPU
- PyTorch 2.9 or a compatible version

Create an environment and install the dependencies:

```bash
conda create -n context2lora python=3.12
conda activate context2lora
bash install.sh
```

`flash-attn` and related CUDA extensions are compiled during installation. Make
sure the CUDA toolkit used for compilation is compatible with your PyTorch
installation. The optional `--use_liger_kernel` training flag additionally
requires the `liger-kernel` package.

## Data format

Training accepts a JSONL file with one object per line:

```json
{"context": "A supporting passage.", "prompt": "A question about it.", "completion": "The answer."}
```

An optional validation file uses the same schema. Data is not included in this
repository.

Large JSONL files can be converted to Arrow before training:

```bash
python scripts/preprocess_arrow.py /path/to/train.jsonl
```

## Training

The launcher defaults to single-node training and reads machine-specific values
from environment variables:

```bash
DATASET_PATH=/path/to/train.jsonl \
EVAL_DATASET_PATH=/path/to/validation.jsonl \
NPROC_PER_NODE=8 \
bash scripts/train.sh
```

Useful overrides include:

- `MODEL_NAME`: Hugging Face model ID (default: `Qwen/Qwen3.5-4B`)
- `PG_CONFIG`: parameter-generator configuration
- `OUTPUT_DIR`: checkpoint and log directory
- `USE_DISTILLATION=0`: run next-token parameter-generator pretraining
- `INIT_WEIGHTS_FROM`: initialize post-distillation from a pretrained checkpoint
- `MAX_SEQ_LEN`, `BATCH_SIZE`, and `GRADIENT_ACCUMULATION_STEPS`
- `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, and `SEED`

For post-distillation from a pretrained parameter generator:

```bash
DATASET_PATH=/path/to/train.jsonl \
INIT_WEIGHTS_FROM=/path/to/pretrain/checkpoints/step_500 \
bash scripts/train.sh
```

Weights & Biases logging is disabled by default. To enable it, authenticate with
W&B using its standard environment-based configuration and set
`ENABLE_WANDB=1`. Never place credentials in source files.

For multi-node runs, invoke `train.py` with a `torchrun` configuration suitable
for your scheduler. Run `python train.py --help` for the complete argument list.

## Inference

`infer.py` reads the model and parameter-generator settings from the
`train_args.json` stored with a training run:

```bash
python infer.py \
  --checkpoint /path/to/output/checkpoints/step_500 \
  --context "The Eiffel Tower was built between 1887 and 1889." \
  --prompt "When was the Eiffel Tower built?"
```

The checkpoint directory must contain the saved parameter-generator weights.

## Evaluation

Evaluation can run during training by passing `--eval_dataset_path` and setting
`--eval_steps` to a positive value. The standalone evaluation helpers in
`evaluate.py` compute token-level precision, recall, and F1.

```bash
python evaluate.py \
  --checkpoint /path/to/output/checkpoints/step_500 \
  --eval_dataset_path /path/to/validation.jsonl \
  --eval_samples 500
```

Results and per-example predictions are written into the checkpoint directory.

## Repository layout

```text
configs/                  Parameter-generator and prompt configurations
scripts/train.sh          Portable single-node training launcher
scripts/preprocess_arrow.py
                          Optional JSONL-to-Arrow preprocessing
src/core/                 Parameter generation and runtime LoRA injection
src/data/                 Dataset registry, formatting, and collation
src/utils/                Distributed, logging, prompt, and PyTorch utilities
train.py                  Training entry point
evaluate.py               Generation and evaluation helpers
infer.py                  Single-example inference
```

## Security

Keep model hub tokens, experiment-tracking keys, storage credentials, private
paths, and infrastructure configuration outside the repository. Supply secrets
through your platform's secret manager or environment variables and review
generated logs and checkpoints before sharing them.

## License

This project is licensed under the [MIT License](LICENSE).
