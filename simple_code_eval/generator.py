import inspect
import json
import os
import warnings
from typing import List

from . import tasks
from .generation import parallel_generations


class Generator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def generate_text(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = (
            min(self.args.limit, len(dataset) - self.args.limit_start)
            if self.args.limit
            else len(dataset)
        )
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [
            task.get_reference(dataset[i])
            for i in range(self.args.limit_start, self.args.limit_start + n_tasks)
        ]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [
                    [task.get_reference(dataset[i], get_solution=True)]
                    for i in range(
                        self.args.limit_start, self.args.limit_start + n_tasks
                    )
                ]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        curr_generations = []  # list[list[str | None] | None]
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)
        intermediate_save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}_intermediate.json"
        curr_sample_idx = len(curr_generations)

        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
            curr_sample_idx=curr_sample_idx,  # curr_sample_idx will added to limit_start to fix indexing
            save_every_k_tasks=self.args.save_every_k_tasks,
            intermediate_generations=curr_generations,
            intermediate_save_generations_path=intermediate_save_generations_path,
        )

        if len(generations[0]) > self.args.n_samples:
            generations = [
                generation[: self.args.n_samples] for generation in generations
            ]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            os.makedirs(os.path.dirname(save_generations_path), exist_ok=True)
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp, indent=2)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            os.makedirs(os.path.dirname(save_references_path), exist_ok=True)
            with open(save_references_path, "w") as fp:
                json.dump(references, fp, indent=2)
                print(f"references were saved at {save_references_path}")
