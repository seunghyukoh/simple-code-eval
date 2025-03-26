import os
from typing import List

from . import tasks

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
    def __init__(self, args):
        self.args = args
        self.allow_code_execution = args.allow_code_execution

    def evaluate(self, task_name: str, generations: List[str], references: List[str]):
        """
        Evaluate the generated code against references.

        Args:
            task_name: Name of the task to evaluate
            generations: List of generated code strings
            references: List of reference code strings

        Returns:
            Results of the evaluation
        """
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        # make sure tokenizer plays nice with multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("Evaluating generations...")
        results = task.process_results(generations, references)
        return results
