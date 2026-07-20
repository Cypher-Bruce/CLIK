from __future__ import annotations

import os

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from src.utils.logging import get_logger
from src.utils.prompts import format_prompt

_DATASET_REGISTRY: dict[str, type[SFTDatasetBuilder]] = {}


def _arrow_cache_path(jsonl_path: str) -> str:
    """Return the Arrow cache directory path for a given JSONL file."""
    return jsonl_path + ".arrow_cache"


def ensure_arrow_cache(jsonl_path: str) -> str:
    """Convert *jsonl_path* to Arrow format if the cache doesn't exist yet.

    Intended to be called on rank 0 only (before a dist.barrier) so that
    the expensive JSON parse happens once.  All ranks then load the
    memory-mapped Arrow cache via :func:`load_arrow_or_jsonl`.

    Returns the Arrow cache directory path.
    """
    cache_dir = _arrow_cache_path(jsonl_path)
    log = get_logger()
    if os.path.isdir(cache_dir):
        log.info(f"Arrow cache already exists: {cache_dir}")
        return cache_dir
    log.info(f"Converting JSONL to Arrow cache: {jsonl_path} -> {cache_dir}")
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    ds.save_to_disk(cache_dir)
    log.info(f"Arrow cache saved ({len(ds)} examples): {cache_dir}")
    return cache_dir


def load_arrow_or_jsonl(jsonl_path: str) -> Dataset:
    """Load from Arrow cache if available, otherwise fall back to JSONL."""
    cache_dir = _arrow_cache_path(jsonl_path)
    log = get_logger()
    if os.path.isdir(cache_dir):
        log.info(f"Loading from Arrow cache: {cache_dir}")
        return load_from_disk(cache_dir)
    log.info(f"Arrow cache not found, loading JSONL directly: {jsonl_path}")
    return load_dataset("json", data_files=jsonl_path, split="train")


def register_dataset(name: str):
    def _register(cls: type[SFTDatasetBuilder]):
        _DATASET_REGISTRY[name] = cls
        return cls

    return _register


class SFTDatasetBuilder:
    """Base class for SFT dataset builders.

    Subclasses implement ``load_raw`` and ``format_example`` which returns::

        {"context": str, "prompt": str, "completion": str}

    The base class handles chat-template formatting and tokenization:

    * In standard mode the user message contains *context + prompt*.
    * In parameter-generation mode (``use_pg=True``) the user message
      contains *prompt only* and the context is tokenized separately as
      ``context_input_ids``.

    Tokenization is lazy — each example is tokenized on fetch in the
    dataloader, avoiding upfront processing and memory overhead.  The
    DataLoader's ``num_workers`` and ``prefetch_factor`` hide the latency.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        use_pg: bool = False,
        use_distillation: bool = False,
        max_samples: int | None = None,
        dataset_path: str | None = None,
        prompt_templates: list[str] | None = None,
        eval_dataset_path: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_pg = use_pg
        self.use_distillation = use_distillation
        self.max_samples = max_samples
        self.dataset_path = dataset_path
        self.prompt_templates = prompt_templates
        self.eval_dataset_path = eval_dataset_path

    # -- override in subclass ---------------------------------------------------

    def load_raw(self) -> Dataset:
        raise NotImplementedError

    def format_example(self, example: dict) -> dict:
        """Return ``{"context": str, "prompt": str, "completion": str}``."""
        raise NotImplementedError

    def download(self):
        """Download dataset files to local cache. Override for remote datasets."""
        pass

    def jsonl_paths(self) -> list[str]:
        """Return JSONL file paths that this builder will load.

        Used by the training script to build Arrow caches on rank 0 before
        the distributed barrier.  Override in subclasses that use hardcoded
        default paths.
        """
        paths = []
        if self.dataset_path:
            paths.append(self.dataset_path)
        if self.eval_dataset_path:
            paths.append(self.eval_dataset_path)
        return paths

    # -- chat message helpers ---------------------------------------------------

    def _build_messages(
        self,
        formatted: dict,
        force_standard: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """Build prompt and completion chat message lists from raw fields.

        When *force_standard* is True, context is always included in the user
        message regardless of ``use_pg``.  Used to build teacher inputs for
        knowledge distillation.  If ``prompt_templates`` is set, the teacher
        message is formatted using a deterministically selected template.
        """
        ctx = formatted.get("context", "")
        prompt = formatted["prompt"]

        if self.prompt_templates:
            if self.use_pg and not force_standard:
                user_content = format_prompt(self.prompt_templates, prompt)
            else:
                user_content = format_prompt(
                    self.prompt_templates,
                    prompt,
                    context=ctx or None,
                )
        elif (self.use_pg and not force_standard) or not ctx:
            user_content = prompt
        else:
            user_content = f"Context: {ctx}\n\nQuestion: {prompt}"

        prompt_msgs = [{"role": "user", "content": user_content}]
        completion_msgs = [{"role": "assistant", "content": formatted["completion"]}]
        return prompt_msgs, completion_msgs

    # -- tokenize a single formatted example ------------------------------------

    def tokenize_example(self, formatted: dict) -> dict | None:
        """Tokenize one formatted example. Returns None if the example is empty."""
        if self.use_pg and "context" not in formatted:
            raise KeyError(
                "Parameter generation mode requires format_example() to return "
                "a 'context' key."
            )

        prompt_msgs, completion_msgs = self._build_messages(formatted)

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        completion_text = self.tokenizer.apply_chat_template(
            prompt_msgs + completion_msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        completion_text = completion_text[len(prompt_text) :]

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(completion_text, add_special_tokens=False)[
            "input_ids"
        ]

        input_ids = (prompt_ids + completion_ids)[: self.max_seq_len]
        completion_mask = ([0] * len(prompt_ids) + [1] * len(completion_ids))[
            : self.max_seq_len
        ]

        if len(input_ids) == 0 or not any(m == 1 for m in completion_mask):
            return None

        result = {"input_ids": input_ids, "completion_mask": completion_mask}

        if self.use_pg:
            result["context_input_ids"] = self.tokenizer(
                formatted["context"], add_special_tokens=False
            )["input_ids"][: self.max_seq_len]

            if self.use_distillation:
                # Teacher tokenization: context included in the user message
                t_prompt_msgs, t_completion_msgs = self._build_messages(
                    formatted,
                    force_standard=True,
                )
                t_prompt_text = self.tokenizer.apply_chat_template(
                    t_prompt_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                t_full_text = self.tokenizer.apply_chat_template(
                    t_prompt_msgs + t_completion_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                t_completion_text = t_full_text[len(t_prompt_text) :]

                t_prompt_ids = self.tokenizer(t_prompt_text, add_special_tokens=False)[
                    "input_ids"
                ]
                t_completion_ids = self.tokenizer(
                    t_completion_text, add_special_tokens=False
                )["input_ids"]

                t_input_ids = (t_prompt_ids + t_completion_ids)[: self.max_seq_len]
                t_completion_mask = (
                    [0] * len(t_prompt_ids) + [1] * len(t_completion_ids)
                )[: self.max_seq_len]

                result["teacher_input_ids"] = t_input_ids
                result["teacher_completion_mask"] = t_completion_mask

        return result

    # -- build methods ----------------------------------------------------------

    def _maybe_truncate(self, raw: Dataset) -> Dataset:
        if self.max_samples is not None and len(raw) > self.max_samples:
            raw = raw.select(range(self.max_samples))
        return raw

    def load_raw_eval(self) -> Dataset | None:
        """Return the validation split, or None if unavailable.

        The default implementation loads from ``eval_dataset_path`` if set.
        Subclasses may override to provide a different eval source (e.g.
        a HuggingFace split).
        """
        if self.eval_dataset_path:
            return load_arrow_or_jsonl(self.eval_dataset_path)
        return None

    def build(self) -> TorchDataset:
        """Return a lazy torch Dataset that tokenizes on each fetch."""
        raw = self._maybe_truncate(self.load_raw())
        return _LazySFTDataset(raw, self)

    def build_eval(self) -> TorchDataset | None:
        """Return a lazy torch Dataset for validation, or None."""
        raw = self.load_raw_eval()
        if raw is None:
            return None
        return _LazySFTDataset(raw, self)


class _LazySFTDataset(TorchDataset):
    """Wraps a raw HuggingFace Dataset and tokenizes on each ``__getitem__``."""

    def __init__(self, raw_dataset: Dataset, builder: SFTDatasetBuilder):
        self.raw = raw_dataset
        self.builder = builder

    def __len__(self):
        return len(self.raw)

    _MAX_FALLBACK_ATTEMPTS = 10

    def __getitem__(self, idx):
        for attempt in range(self._MAX_FALLBACK_ATTEMPTS):
            candidate = (idx - attempt - 1) % len(self.raw) if attempt else idx
            formatted = self.builder.format_example(self.raw[candidate])
            result = self.builder.tokenize_example(formatted)
            if result is not None:
                return result
        raise RuntimeError(
            f"Failed to tokenize example at index {idx} and "
            f"{self._MAX_FALLBACK_ATTEMPTS} neighbors."
        )


def build_data_collator(tokenizer, use_pg: bool = False):
    """Return the appropriate data collator.

    When *use_pg* is True the collator also pads ``context_input_ids`` and
    produces ``context_attention_mask`` needed by the parameter-generation
    training loop.
    """
    base_collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )
    if not use_pg:
        return base_collator

    pad_id = tokenizer.pad_token_id

    def _pg_collator(features):
        context_seqs = [f.pop("context_input_ids") for f in features]
        teacher_seqs = [f.pop("teacher_input_ids", None) for f in features]
        teacher_comp_masks = [f.pop("teacher_completion_mask", None) for f in features]
        # Save completion_mask before base_collator consumes it for label creation
        student_comp_masks = [list(f["completion_mask"]) for f in features]
        batch = base_collator(features)

        max_ctx_len = max(len(s) for s in context_seqs)
        padded_ctx = []
        ctx_masks = []
        for seq in context_seqs:
            pad_len = max_ctx_len - len(seq)
            padded_ctx.append(seq + [pad_id] * pad_len)
            ctx_masks.append([1] * len(seq) + [0] * pad_len)

        batch["context_input_ids"] = torch.tensor(padded_ctx, dtype=torch.long)
        batch["context_attention_mask"] = torch.tensor(ctx_masks, dtype=torch.long)

        seq_len = batch["input_ids"].shape[1]
        padded_s_masks = []
        for m in student_comp_masks:
            padded_s_masks.append(m + [0] * (seq_len - len(m)))
        batch["completion_mask"] = torch.tensor(padded_s_masks, dtype=torch.long)

        if any(s is not None for s in teacher_seqs):
            valid_t_seqs = [s for s in teacher_seqs if s is not None]
            max_t_len = max(len(s) for s in valid_t_seqs)
            padded_t, t_attn_masks, t_comp_masks = [], [], []
            for seq, cmask in zip(teacher_seqs, teacher_comp_masks):
                if seq is None:
                    padded_t.append([pad_id] * max_t_len)
                    t_attn_masks.append([0] * max_t_len)
                    t_comp_masks.append([0] * max_t_len)
                else:
                    pad_len = max_t_len - len(seq)
                    padded_t.append(seq + [pad_id] * pad_len)
                    t_attn_masks.append([1] * len(seq) + [0] * pad_len)
                    t_comp_masks.append(cmask + [0] * pad_len)

            batch["teacher_input_ids"] = torch.tensor(padded_t, dtype=torch.long)
            batch["teacher_attention_mask"] = torch.tensor(
                t_attn_masks, dtype=torch.long
            )
            batch["teacher_completion_mask"] = torch.tensor(
                t_comp_masks, dtype=torch.long
            )

        return batch

    return _pg_collator


def _auto_register():
    """Import every .py file in src/data/ so @register_dataset decorators run."""
    import importlib
    import pathlib

    pkg_dir = pathlib.Path(__file__).parent
    for f in pkg_dir.glob("*.py"):
        if f.name == "__init__.py":
            continue
        importlib.import_module(f"src.data.{f.stem}")


def _get_builder_cls(dataset_name: str):
    _auto_register()
    if dataset_name not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown dataset {dataset_name!r}. Available: {available}")
    return _DATASET_REGISTRY[dataset_name]


def download_sft_dataset(dataset_name: str):
    """Download dataset files to local cache (call on rank 0 before barrier)."""
    cls = _get_builder_cls(dataset_name)
    cls(tokenizer=None, max_seq_len=0).download()


def prepare_sft_dataset(
    dataset_name: str,
    dataset_path: str | None = None,
    eval_dataset_path: str | None = None,
):
    """Download and build Arrow caches for all JSONL files (rank 0 only).

    Call before the distributed barrier so that other ranks can instantly
    memory-map the cached Arrow tables.
    """
    cls = _get_builder_cls(dataset_name)
    builder = cls(
        tokenizer=None,
        max_seq_len=0,
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
    )
    builder.download()
    for path in builder.jsonl_paths():
        ensure_arrow_cache(path)


def build_sft_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_len: int,
    use_pg: bool = False,
    use_distillation: bool = False,
    max_samples: int | None = None,
    dataset_path: str | None = None,
    prompt_templates: list[str] | None = None,
):
    builder = _get_builder_cls(dataset_name)(
        tokenizer,
        max_seq_len,
        use_pg=use_pg,
        use_distillation=use_distillation,
        max_samples=max_samples,
        dataset_path=dataset_path,
        prompt_templates=prompt_templates,
    )
    return builder.build()


def build_eval_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_len: int,
    use_pg: bool = False,
    dataset_path: str | None = None,
    eval_dataset_path: str | None = None,
    prompt_templates: list[str] | None = None,
):
    builder = _get_builder_cls(dataset_name)(
        tokenizer,
        max_seq_len,
        use_pg=use_pg,
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
        prompt_templates=prompt_templates,
    )
    return builder.build_eval()
