import inspect
import json
import os
import warnings
from typing import List

from . import tasks
from .generator import Generator

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

        # Initialize generator
        self.generator = Generator(accelerator, model, tokenizer, args)

    def evaluate(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generator.generate_text(
            task_name, intermediate_generations=intermediate_generations
        )

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}.json"
                save_references_path = f"{os.path.splitext(self.args.save_references_path)[0]}_{task_name}.json"
                self.generator.save_json_files(
                    generations,
                    references,
                    save_generations_path,
                    save_references_path,
                )

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task.process_results(generations, references)
            return results
