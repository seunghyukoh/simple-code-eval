# Simple Code Eval

A simple code evaluation package that helps you evaluate code using OpenAI's GPT models. This project is inspired by [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness).

## Installation

You can install the package using pip:

```bash
git clone https://github.com/seunghyukoh/simple-code-eval.git
cd simple-code-eval
pip install -e .
```

## Usage

```python
from simple_code_eval import CodeEvaluator

# Initialize the evaluator
evaluator = CodeEvaluator()

# Evaluate code
result = evaluator.evaluate_code("your_code_here")
print(result)
```

## Author

Created by Seunghyuk Oh

## License

This project is licensed under the MIT License - see the LICENSE file for details.
