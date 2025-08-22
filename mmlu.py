from typing import Any
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from detection.task_list import TASKS


class MMLU(Dataset):
    def __init__(
        self,
        task,
        cache_dir: str,
        split: str = "train",
        options: list[str] = ["A", "B", "C", "D"],
        option_separator: str = ")",
        postfix_token: int = None,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        assert task in TASKS
        assert len(options) == num_classes
        assert split in ["train", "validation", "test"]

        self.task = task
        self.split = split
        self.options = options
        self.option_separator = option_separator
        if postfix_token is not None:
            self.postfix_token = torch.ones((1,), dtype=torch.long) * postfix_token
        else:
            self.postfix_token = postfix_token

        self.dataset = load_dataset(
            "lukaemon/mmlu",
            self.task,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.idx_to_class: dict[int, str] = {
            i: name for i, name in enumerate(self.options)
        }
        self.class_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.options)
        }
        self.target_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(["A", "B", "C", "D"])
        }

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, index) -> Any:
        data = self.dataset[self.split][index]
        text = (
            data["input"]
            + f"\n{self.options[0]}{self.option_separator} "
            + data["A"]
            + f"\n{self.options[1]}{self.option_separator} "
            + data["B"]
            + f"\n{self.options[2]}{self.option_separator} "
            + data["C"]
            + f"\n{self.options[3]}{self.option_separator} "
            + data["D"]
            + "\n"
        )
        return {
            "text": text,
            "label": int(self.target_to_idx[data["target"]]),
            "task": self.task.replace("_", " "),
        }
    
    
class MMLUPro(Dataset):
    def __init__(self, subject, cache_dir: str, split: str = "validation"):
        super().__init__()
        assert split in ["train", "validation", "test"]

        self.subject = subject
        self.split = split
        self.options = ["A","B","C","D","E","F","G","H","I","J"]
        self.num_classes = len(self.options)

        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split, cache_dir=cache_dir)
        self.dataset = ds.filter(lambda x: x["subject"] == subject)

        self.target_to_idx = {name: i for i, name in enumerate(self.options)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        text = data["input"]
        for opt in self.options:
            text += f"\n{opt}) {data[opt]}"
        return {
            "text": text,
            "label": self.target_to_idx[data["target"]],
            "task": self.subject.replace("_", " "),
        }


if __name__ == "__main__":
    import json
    import os

    # Define the tasks to be processed
    target_tasks = TASKS 

    # Define the cache directory and the save directory
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir = "/data2/paveen/RolePlaying/src/models/components/mmlupro"
    os.makedirs(save_dir, exist_ok=True)

    for task in target_tasks:
        print(f"=== Processing task: {task.replace('_', ' ')} ===")
        
        # sc = MMLU(task, cache_dir=cache_dir, split="test")
        sc = MMLUPro(task, cache_dir=cache_dir, split="test")
    
        task_data = []
        total_samples = len(sc)
        print(f"Total samples in {task}: {total_samples}")

        for i in range(total_samples):
            sample = sc[i]
            task_data.append({
                "task": sample["task"].replace('_', ' '),
                "text": sample["text"],
                "label": sample["label"]
            })

        save_path = os.path.join(save_dir, f"{task}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(task_data, f, ensure_ascii=False, indent=4)
        print(f"\nData for task '{task}' saved to {save_path}\n")

    print("\n=== All target tasks have been processed and saved successfully! ===\n")
