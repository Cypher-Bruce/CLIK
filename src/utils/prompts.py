from __future__ import annotations

import hashlib
import json


def load_prompt_templates(path: str) -> list[str]:
    """Load a prompt config JSON -- a plain list of template strings.

    Each template must contain an ``{input}`` placeholder for the question.
    """
    with open(path) as f:
        templates = json.load(f)
    if not isinstance(templates, list) or len(templates) == 0:
        raise ValueError(
            f"Prompt config {path} must be a non-empty JSON list of template strings."
        )
    return templates


def _stable_hash(s: str) -> int:
    """Platform-independent deterministic hash for strings.

    Python's built-in ``hash()`` is randomized via PYTHONHASHSEED and can
    differ across runs, Python versions, and platforms.  MD5 is stable
    everywhere.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def format_prompt(
    templates: list[str],
    prompt: str,
    context: str | None = None,
) -> str:
    """Deterministically select a template and format it.

    The same prompt always selects the same template, ensuring consistency
    between offline generation and training.  Context is prepended only
    when provided (teacher / generation path), so the template itself
    works for both student (no context) and teacher (with context).
    """
    idx = _stable_hash(prompt) % len(templates)
    text = templates[idx].format(input=prompt)
    if context:
        text = f"Context: {context}\n\n{text}"
    return text
