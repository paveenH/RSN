import logging
from typing import Any, Dict, List

import os
import spacy
import torch
from lightning import LightningModule
from torch.nn import ModuleDict
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np

# A logger for this file
log = logging.getLogger(__name__)

class LanguageTaskOnTheFlyLitModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        llm,
        data_path: str,
        num_classes: int,
        seed: int,
        characters: List[str] = ["2 year old", "4 year old"],
        template: str = "You are a {character}, You are a {character}, You are a {character}, would you answer the following question with A, B, C or D? \n Question: {context}\n Answer: ",
        max_tries: int = 10,
        extract_hidden: bool = False, 
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.characters = characters
        self.llm = llm
        self.template = template
        self.seed = seed
        self.num_classes = num_classes
        self.data_path = data_path
        self.max_tries = max_tries
        self.extract_hidden = extract_hidden

        log.info(f"Template: {self.template}")

        log.info("Loading spaCy for sentence cleaning")
        self.nlp = spacy.load("en_core_web_sm")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["llm", "model"])
        
        # Storage structure for saving hidden states
        if self.extract_hidden:
            self.hidden_states_storage: Dict[str, List[np.ndarray]] = {
                character: [] for character in self.characters
            }
            self.test_task = ""
        else:
            self.train_accs = ModuleDict(
                {
                    character: Accuracy(task="multiclass", num_classes=num_classes)
                    for character in characters
                }
            )
            self.val_accs = ModuleDict(
                {
                    character: Accuracy(task="multiclass", num_classes=num_classes)
                    for character in characters
                }
            )
            self.test_accs = ModuleDict(
                {
                    character: Accuracy(task="multiclass", num_classes=num_classes)
                    for character in characters
                }
            )
            
            self.train_losses = ModuleDict(
                {character: MeanMetric() for character in characters}
            )
            
            self.test_losses = ModuleDict(
                {character: MeanMetric() for character in characters}
            )

            # for tracking best so far validation accuracy
            self.val_acc_bests = ModuleDict(
                {character: MaxMetric() for character in characters}
            )
        

        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def module_step_llama(self, batch: dict, batch_idx: int):
        texts = batch["text"]
        labels = batch["label"]
        task = list(set(batch["task"]))
        assert len(task) == 1, "Each batch should contain only one unique task."
        task = task[0]
        
        # Get an ordered list of answers ["A", "B", "C", "D"]
        ordered_answers = [
            self.trainer.datamodule.data_test.idx_to_class[i]
            for i in range(self.num_classes)
        ]

        return_values = {}
        for character in self.characters:
            # Generate the prompts
            prompts = [
                self.template.format(character=character, context=t)
                for t in texts
            ]

            # Generate answers using the LLM
            generated_outputs = self.llm.generate(prompts, max_new_tokens=1)  # Assuming max_new_tokens=1 for single-token output
            
            # Loop through each output to check if it matches the expected answers
            pred_classes = []
            invalid_count = 0
            for output in generated_outputs:
                # Check if the output is one of the ordered answers
                output_stripped = ''.join(filter(str.isalpha, output.strip())).upper()
                if output_stripped in ordered_answers:
                    pred_class = ordered_answers.index(output_stripped)
                else:
                    default_answer = "C"
                    if default_answer in ordered_answers:
                        pred_class = ordered_answers.index(default_answer)  # default C
                        invalid_count += 1
                        log.warning(f"Missing answer: {output}. Defaulted to '{default_answer}'.")
                    else:
                        log.error(f"Default answer '{default_answer}' not found in ordered_answers.")
                        pred_class = 0  # Fallback to first class or handle appropriately
                if invalid_count > 0:
                    log.warning(f"Character {character}: {invalid_count} invalid answers generated, defaulted to '{default_answer}'.")
                pred_classes.append(pred_class)

            # Convert to tensors for further processing
            pred_classes = torch.tensor(pred_classes).to(self.device)
            labels_on_device = labels.long()

            return_values[character] = {
                "pred_classes": pred_classes,
                "labels": labels_on_device,
            }

        return return_values
    
    def module_step_llama_hidden(self, batch: dict, batch_idx: int):
        texts = batch["text"]
        task = list(set(batch["task"]))
        assert len(task) == 1, "Each batch should contain only one unique task."
        task = task[0]
        
        self.test_task = task

        return_values = {}
        for character in self.characters:
            # Generate the prompts
            prompts = [
                self.template.format(character=character, context=t)
                for t in texts
            ]

            # Extract hidden state
            for idx, prompt in enumerate(prompts):
                hidden_states = self.llm.get_hidden_states(
                    prompt=prompt,
                    character=character
                )
                
                if any(pos is None for pos in hidden_states):
                    log.warning(f"Some positions not found for prompt index {idx}.")
                    continue
                else:
                    pos_arrays = []
                    for pos in hidden_states:
                        pos_array = np.stack(pos, axis=0) 
                        pos_arrays.append(pos_array)
                    
                    hidden_states_array = np.stack(pos_arrays, axis=0)
                    self.hidden_states_storage[character].append(hidden_states_array)
                        
        return return_values
    
    
    def module_step(self, batch: dict, batch_idx: int):  
        if hasattr(self.llm, 'model_path') and "llama3" in self.llm.model_path.lower():
            if getattr(self, 'extract_hidden', False):
                return self.module_step_llama_hidden(batch, batch_idx)
            else:
                return self.module_step_llama(batch, batch_idx)

        # one method to do it all
        text = batch["text"]
        label = batch["label"]
        task = list(set(batch["task"]))
        assert len(task) == 1, "Each batch should contain only one unique task."
        task = task[0]

        # obtain ordered list of descriptions
        ordered_answers = [
            self.trainer.datamodule.data_test.idx_to_class[i]
            for i in range(self.num_classes)
        ]
        target_tokens = self.llm.tokenizer(ordered_answers, return_tensors="pt")
        # target_tokens = target_tokens.input_ids[:, -1].to(label.device)

        # now we want to run this for each character
        return_values = {}
        for character in self.characters:
            prompts = [
                self.template.format(character=character, context=t, task=task)
                for t in text
            ]
            if (
                hasattr(self.trainer.datamodule.data_test, "postfix_token")
                and self.trainer.datamodule.data_test.postfix_token is not None
            ):
                logits = self.llm.get_logits(
                    prompts,
                    postfix_token=self.trainer.datamodule.data_test.postfix_token,
                )
            else:
                logits = self.llm.get_logits(prompts)
                
            target_tokens_device = target_tokens.input_ids[:, -1].to(logits.device).long()
            logits_per_class = logits[:, -1, target_tokens_device]

            # we can take the softmax to get the label probabilities
            probs = logits_per_class.softmax(dim=1)
            pred_classes = probs.argmax(dim=1)

            # compute the loss and return it
            label_on_device = label.to(logits.device).long()
            loss = self.criterion(probs, label_on_device)

            return_values[character] = {
                "loss": loss,
                "probs": probs,
                "pred_classes": pred_classes,
            }

        return return_values


    def test_step(self, batch: dict, batch_idx: int):
        label = batch["label"]
        out = self.module_step(batch, batch_idx)
        
        # When extracting hidden states, no accuracy calculation is performed
        if self.extract_hidden:                
            return out

        for character, results in out.items():
            if "llama3" in self.llm.model_path.lower():
                pred_classes = results["pred_classes"]
                label = results["labels"]
                self.test_accs[character](pred_classes, label)
            else:
                loss = results["loss"]
                probs = results["probs"]
                pred_classes = results["pred_classes"]
                
                loss = loss.to(self.device)
                probs = probs.to(self.device)
                pred_classes = pred_classes.to(self.device)
                label = label.to(self.device)

                self.test_losses[character](loss)
                self.test_accs[character](probs, label)

                self.log(
                    f"test/{self.trainer.datamodule.data_test.task}/{character}/loss",
                    self.test_losses[character],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            self.log(
                f"test/{self.trainer.datamodule.data_test.task}/{character}/acc",
                self.test_accs[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out
    
    def on_test_epoch_end(self):
        if self.extract_hidden:
            task_name = self.test_task
            # Save the hidden state as a .npy file
            for character, hidden_states in self.hidden_states_storage.items():
                hidden_states_array = np.array(hidden_states)
                save_dir = os.path.join(self.data_path, f"hidden_states_{task_name}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{character.replace(' ', '_')}_hidden_states.npy")
                np.save(save_path, hidden_states_array)
                log.info(f"Saved hidden states for {character} to {save_path}")

            # Clear storage
            self.hidden_states_storage = {
                character: [] for character in self.characters
            }
        else:
            # Original accuracy related processing
            metric_dict = {}
            for character in self.characters:
                acc = self.test_accs[character].compute()
                metric_dict[f"test_acc_{character}"] = acc
                self.test_accs[character].reset()
            
            return metric_dict