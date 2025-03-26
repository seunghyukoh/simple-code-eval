import inspect
from pprint import pprint

from . import humaneval, mbpp, mbppplus
from .base import Task

__all__ = ["Task"]

TASK_REGISTRY = {
    "mbpp": mbpp.MBPP,
    "mbppplus": mbppplus.MBPPPlus,
    **humaneval.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
