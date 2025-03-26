import fnmatch
import json
import os
import warnings

import datasets
import torch
import transformers
from accelerate import Accelerator
from simple_code_eval.arguments import EvalArguments
from simple_code_eval.evaluator import Evaluator
from simple_code_eval.generator import Generator
from simple_code_eval.tasks import ALL_TASKS
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--left_padding",
        action="store_true",
        help="Force left padding, needed for models like chatglm3-6b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--save_every_k_tasks",
        type=int,
        default=-1,
        help="Optional saving after every k tasks",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--load_generations_intermediate_paths",
        type=str,
        nargs="*",
        help="List of paths for saving the intermediate code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the references solutions/tests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    args = parser.parse_args()

    assert not args.save_generations or (
        args.save_generations and args.save_generations_path is not None
    ), "save_generations_path is required when save_generations is True"
    assert not args.save_references or (
        args.save_references and args.save_references_path is not None
    ), "save_references_path is required when save_references is True"

    return args


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    accelerator = Accelerator()

    # Initialize model and tokenizer
    kwargs = {}
    if args.use_auth_token:
        kwargs["use_auth_token"] = True
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if args.revision:
        kwargs["revision"] = args.revision

    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    if args.modeltype == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **kwargs,
            torch_dtype=torch.float16 if args.precision == "fp16" else torch.float32,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            device_map="auto",
            max_memory=get_gpus_max_memory(
                args.max_memory_per_gpu, accelerator.num_processes
            )
            if args.max_memory_per_gpu
            else None,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model,
            **kwargs,
            torch_dtype=torch.float16 if args.precision == "fp16" else torch.float32,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            device_map="auto",
            max_memory=get_gpus_max_memory(
                args.max_memory_per_gpu, accelerator.num_processes
            )
            if args.max_memory_per_gpu
            else None,
        )

    # Load PEFT model if specified
    if args.peft_model:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.peft_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize Generator and Evaluator
    generator = Generator(accelerator, model, tokenizer, args)
    evaluator = Evaluator(args)

    # Process each task
    for task_name in task_names:
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}\n")

        # Generate code if not loading from file
        if not args.load_generations_path:
            generations, references = generator.generate_text(task_name)

            # Save generations and references if requested
            if accelerator.is_main_process:
                if args.save_generations:
                    save_generations_path = f"{os.path.splitext(args.save_generations_path)[0]}_{task_name}.json"
                    generator.save_json_files(
                        generations,
                        references,
                        save_generations_path,
                        f"{os.path.splitext(args.save_references_path)[0]}_{task_name}.json"
                        if args.save_references
                        else None,
                    )
        else:
            # Load generations from file
            with open(args.load_generations_path, "r") as f:
                generations = json.load(f)
            references = None  # References will be loaded by the task

        # Evaluate if not generation only
        if not args.generation_only:
            results = evaluator.evaluate(task_name, generations, references)

            # Save results
            if accelerator.is_main_process:
                print(results)

                with open(args.metric_output_path, "w") as f:
                    json.dump({task_name: results, "config": vars(args)}, f, indent=2)
                    f.write("\n")
                print(f"Results saved to {args.metric_output_path}")


if __name__ == "__main__":
    main()
