from datasets import Dataset

from src.data import SFTDatasetBuilder, load_arrow_or_jsonl, register_dataset


@register_dataset("fineweb_edu_qa")
class FinewebEduQASFTDatasetBuilder(SFTDatasetBuilder):
    """Load a user-provided FineWeb-Edu QA JSONL or Arrow dataset."""

    def _require_dataset_path(self) -> str:
        if not self.dataset_path:
            raise ValueError(
                "fineweb_edu_qa requires --dataset_path pointing to a JSONL "
                "file or preprocessed Arrow directory."
            )
        return self.dataset_path

    def load_raw(self) -> Dataset:
        return load_arrow_or_jsonl(self._require_dataset_path())

    def jsonl_paths(self) -> list[str]:
        paths = [self._require_dataset_path()]
        if self.eval_dataset_path:
            paths.append(self.eval_dataset_path)
        return paths

    def format_example(self, example: dict) -> dict:
        return {
            "context": example["context"],
            "prompt": example["prompt"],
            "completion": example["completion"],
        }
