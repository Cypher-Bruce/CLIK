import os

from datasets import Dataset, load_dataset, load_from_disk

from src.data import SFTDatasetBuilder, register_dataset

_LOCAL_CACHE = "/tmp/c2l_datasets/squad"
_LOCAL_CACHE_VAL = "/tmp/c2l_datasets/squad_validation"


@register_dataset("squad")
class SquadSFTDatasetBuilder(SFTDatasetBuilder):
    def download(self):
        if not os.path.isdir(_LOCAL_CACHE):
            ds = load_dataset("squad", split="train", trust_remote_code=True)
            ds.save_to_disk(_LOCAL_CACHE)
        if not os.path.isdir(_LOCAL_CACHE_VAL):
            ds = load_dataset("squad", split="validation", trust_remote_code=True)
            ds.save_to_disk(_LOCAL_CACHE_VAL)

    def load_raw(self) -> Dataset:
        return load_from_disk(_LOCAL_CACHE)

    def load_raw_eval(self) -> Dataset | None:
        if os.path.isdir(_LOCAL_CACHE_VAL):
            return load_from_disk(_LOCAL_CACHE_VAL)
        return None

    def format_example(self, example: dict) -> dict:
        return {
            "context": example["context"],
            "prompt": example["question"],
            "completion": example["answers"]["text"][0],
        }
