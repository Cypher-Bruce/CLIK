#!/usr/bin/env python3
"""Offline preprocessing: convert JSONL dataset files to Arrow caches.

This replicates the on-first-run conversion that ``prepare_sft_dataset``
performs inside ``train.py`` on rank 0, but as a standalone script you can
run ahead of time (e.g. on a CPU node or in a prep job) so that training
starts instantly without the JSON-parse overhead.

Usage
-----
Convert explicit JSONL files::

    python scripts/preprocess_arrow.py file1.jsonl file2.jsonl

Convert all JSONL files that a registered dataset builder will load::

    python scripts/preprocess_arrow.py --dataset jsonl --dataset-path /data/train.jsonl
    python scripts/preprocess_arrow.py --dataset fineweb_edu_qa
    python scripts/preprocess_arrow.py --dataset jsonl \
        --dataset-path /data/train.jsonl --eval-dataset-path /data/eval.jsonl

Force rebuild even if the Arrow cache already exists::

    python scripts/preprocess_arrow.py --force file1.jsonl

Both modes can be combined in a single invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pyarrow as pa
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from src.data import _arrow_cache_path, _get_builder_cls

ARROW_BATCH_SIZE = 10_000


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} PB"


def _read_jsonl_to_arrow(jsonl_path: str) -> Dataset:
    """Read a JSONL file into a HF Dataset with a real byte-level progress bar."""
    file_size = os.path.getsize(jsonl_path)
    basename = os.path.basename(jsonl_path)

    rows: list[dict] = []
    batches: list[pa.RecordBatch] = []
    schema: pa.Schema | None = None

    with (
        open(jsonl_path, "r") as f,
        tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Reading {basename}",
        ) as pbar,
    ):
        for line in f:
            pbar.update(len(line.encode("utf-8")))
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= ARROW_BATCH_SIZE:
                tbl = pa.Table.from_pylist(rows)
                if schema is None:
                    schema = tbl.schema
                batches.append(tbl.to_batches()[0])
                rows.clear()

    if rows:
        tbl = pa.Table.from_pylist(rows)
        if schema is None:
            schema = tbl.schema
        batches.append(tbl.to_batches()[0])
        rows.clear()

    if not batches:
        return Dataset.from_dict({})

    table = pa.Table.from_batches(batches, schema=schema)
    return Dataset(table)


def convert_one(jsonl_path: str, *, force: bool = False) -> None:
    """Convert a single JSONL file to Arrow cache."""
    if not os.path.isfile(jsonl_path):
        logger.error(f"File not found: {jsonl_path}")
        return

    cache_dir = _arrow_cache_path(jsonl_path)

    if os.path.isdir(cache_dir):
        if not force:
            logger.info(f"Arrow cache already exists (skip): {cache_dir}")
            return
        logger.info(f"Removing existing Arrow cache (--force): {cache_dir}")
        shutil.rmtree(cache_dir)

    file_size = os.path.getsize(jsonl_path)
    t0 = time.perf_counter()

    ds = _read_jsonl_to_arrow(jsonl_path)

    logger.info(f"Saving Arrow cache ({len(ds)} rows) ...")
    ds.save_to_disk(cache_dir)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Done: {len(ds)} examples, {_human_size(file_size)}, {elapsed:.1f}s -> {cache_dir}"
    )


def collect_paths_from_dataset(
    dataset_name: str,
    dataset_path: str | None,
    eval_dataset_path: str | None,
) -> list[str]:
    """Instantiate a dataset builder and return its JSONL paths."""
    cls = _get_builder_cls(dataset_name)
    builder = cls(
        tokenizer=None,
        max_seq_len=0,
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
    )
    builder.download()
    return builder.jsonl_paths()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JSONL files to Arrow caches for fast training startup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="JSONL files to convert directly.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Registered dataset name (e.g. 'jsonl', 'fineweb_edu_qa'). "
        "Converts all JSONL files the builder would load.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override dataset_path for the builder (like --dataset_path in train.py).",
    )
    parser.add_argument(
        "--eval-dataset-path",
        type=str,
        default=None,
        help="Override eval_dataset_path for the builder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild Arrow caches even if they already exist.",
    )
    args = parser.parse_args()

    paths: list[str] = list(args.files)

    if args.dataset:
        builder_paths = collect_paths_from_dataset(
            args.dataset, args.dataset_path, args.eval_dataset_path
        )
        logger.info(
            f"Dataset {args.dataset!r} resolved to {len(builder_paths)} JSONL path(s): {builder_paths}"
        )
        paths.extend(builder_paths)

    if not paths:
        parser.error(
            "Nothing to do. Provide JSONL files as positional args "
            "and/or use --dataset."
        )

    seen: set[str] = set()
    unique_paths = []
    for p in paths:
        p = os.path.abspath(p)
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    logger.info(f"Will process {len(unique_paths)} file(s)")
    t_total = time.perf_counter()

    for i, p in enumerate(unique_paths, 1):
        logger.info(f"[{i}/{len(unique_paths)}] {p}")
        convert_one(p, force=args.force)

    elapsed = time.perf_counter() - t_total
    logger.info(f"All done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
