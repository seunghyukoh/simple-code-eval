"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems,
designed to be solvable by entry level programmers, covering programming fundamentals,
standard library functionality, and so on. Each problem consists of a task description,
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

from typing import Callable

from .base import Task
from .custom_metrics.code_eval import compute_code_eval

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    def __init__(self, apply_chat_template: Callable):
        super().__init__(
            stop_words=[
                "\nclass",
                "\nassert",
                '\n"""',
                "\nprint",
                "\nif",
                "\n<|/",
                "\n```",
                "\n[DONE]",
            ],
            requires_execution=True,
        )

        assert apply_chat_template is not None, "apply_chat_template must be provided"
        self.apply_chat_template = apply_chat_template

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    @property
    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""

        raw_examples = [
            {
                "text": "Write a function to find the similar elements from the given two tuple lists.",
                "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
                "task_id": 2,
                "test_setup_code": "",
                "test_list": [
                    "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                    "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                    "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
                ],
                "challenge_test_list": [],
            },
            {
                "text": "Write a python function to identify non-prime numbers.",
                "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
                "task_id": 3,
                "test_setup_code": "",
                "test_list": [
                    "assert is_not_prime(2) == False",
                    "assert is_not_prime(10) == True",
                    "assert is_not_prime(35) == True",
                ],
                "challenge_test_list": [],
            },
            {
                "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
                "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
                "task_id": 4,
                "test_setup_code": "",
                "test_list": [
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
                ],
                "challenge_test_list": [],
            },
        ]
        messages = []
        for raw_message in raw_examples:
            # User
            description = raw_message["text"]
            reference = "\n".join(raw_message["test_list"])
            user_content = f"Here is your task: {description}\n\nYour code should pass these tests:\n\n```Python\n{reference}\n```\n\nLet's think step by step. Put your final answer at the end with\n\n[BEGIN]\n{{your_code}}\n[DONE]"

            # Assistant
            assistant_content = f"""[BEGIN]\n{raw_message["code"]}\n[DONE]"""

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

        return messages

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["text"]
        reference = self.get_reference(doc)

        user_content = f"Here is your task: {description}\n\nYour code should pass these tests:\n\n```Python\n{reference}\n```\n\nLet's think step by step. Put your final answer at the end with\n\n[BEGIN]\n{{your_code}}\n[DONE]"
        messages = [
            {
                "role": "system",
                "content": "You are an expert Python programmer.",
            },
            *self.fewshot_examples,
            {"role": "user", "content": user_content},
        ]
        prompt = self.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])

    def get_solution(self, doc):
        """Builds the solution for the doc."""
        return doc["code"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references, num_workers=4):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :param num_workers: int
            number of worker threads to use for parallel evaluation
        """
        pass_at_k, results = compute_code_eval(
            references=references, predictions=generations, num_workers=num_workers
        )
        return pass_at_k, results


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    task = MBPP(apply_chat_template=tokenizer.apply_chat_template)
    print(task.get_prompt(task.dataset["test"][0]))
