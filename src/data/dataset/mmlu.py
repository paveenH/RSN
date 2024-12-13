from typing import Any
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

TASKS = [
"high_school_european_history",
"business_ethics",
"clinical_knowledge",
"medical_genetics",
"high_school_us_history",
"high_school_physics",
"high_school_world_history",
"virology",
"high_school_microeconomics",
"econometrics",
"college_computer_science",
"high_school_biology",
"abstract_algebra",
"professional_accounting",
"philosophy",
"professional_medicine",
"nutrition",
"global_facts",
"machine_learning",
"security_studies",
"public_relations",
"professional_psychology",
"prehistory",
"anatomy",
"human_sexuality",
"college_medicine",
"high_school_government_and_politics",
"college_chemistry",
"logical_fallacies",
"high_school_geography",
"elementary_mathematics",
"human_aging",
"college_mathematics",
"high_school_psychology",
"formal_logic",
"high_school_statistics",
"international_law",
"high_school_mathematics",
"high_school_computer_science",
"conceptual_physics",
"miscellaneous",
"high_school_chemistry",
"marketing",
"professional_law",
"management",
"college_physics",
"jurisprudence",
"world_religions",
"sociology",
"us_foreign_policy",
"high_school_macroeconomics",
"computer_security",
"moral_scenarios",
"moral_disputes",
"electrical_engineering",
"astronomy",
"college_biology",
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
            print(f"\nSample {i+1}/{total_samples}:")
            print(f"Task Name: {sample['task']}")
            print(f"Text:\n{sample['text']}")
            print(f"Label: {sample['label']}")

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
