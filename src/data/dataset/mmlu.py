from typing import Any
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_medicine",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


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


if __name__ == "__main__":
    import json
    import os

    # Define the tasks to be processed
    target_tasks = ["management", "medical_genetics"]

    # Define the cache directory and the save directory
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir = "/data2/paveen/RolePlaying/src/models/components/mmlu"
    os.makedirs(save_dir, exist_ok=True)

    for task in target_tasks:
        print(f"=== Processing task: {task.replace('_', ' ')} ===")
        
        sc = MMLU(task, cache_dir=cache_dir, split="test")
    
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
