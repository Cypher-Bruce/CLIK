from datasets import Dataset

from src.data import SFTDatasetBuilder, load_arrow_or_jsonl, register_dataset


@register_dataset("jsonl")
class JsonlSFTDatasetBuilder(SFTDatasetBuilder):
    """Generic JSONL dataset with {context, prompt, completion} fields.

    Requires ``dataset_path`` to be set (passed via ``--dataset_path`` in
    training).
    """

    def load_raw(self) -> Dataset:
        if not self.dataset_path:
            raise ValueError(
                "JsonlSFTDatasetBuilder requires --dataset_path pointing to a JSONL file."
            )
        return load_arrow_or_jsonl(self.dataset_path)

    def load_raw_eval(self) -> Dataset | None:
        if not self.eval_dataset_path:
            return None
        return load_arrow_or_jsonl(self.eval_dataset_path)

    def format_example(self, example: dict) -> dict:
        return {
            "context": example["context"],
            "prompt": example["prompt"],
            "completion": example["completion"],
        }
