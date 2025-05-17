# Python Test Generator

A tool for automatically generating unit tests from Python source code using LLM.

## Overview

This project leverages Large Language Models to analyze Python source code and automatically generate appropriate unit tests. It supports various models, metrics collection, and fine-tuning to create high-quality test suites for Python projects.

## Features

- **Code Analysis**: Extract functions and methods from Python source code
- **Test Generation**: Generate pytest-based unit tests using local or remote LLMs
- **Test Execution**: Automatically run generated tests and evaluate coverage
- **Metric Collection**: Calculate test quality metrics (CodeBLEU, Pass@k, Syntax Validity)
- **Model Comparison**: Compare the performance of different LLMs for test generation
- **Fine-tuning**: Train models on code-test pairs to improve generation quality

## Installation

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Generating Tests

```bash
python generate_tests.py --source path/to/your/file.py --output tests/
```

Options:
- `--source`: Path to the source file
- `--output`: Directory to save generated tests (default: `tests/`)
- `--model`: Model to use for generation (default: "deepseek-ai/deepseek-coder-1.3b-instruct")
- `--cpu-only`: Force CPU execution even if GPU is available

### Collecting Metrics

```bash
python calculate_metrics.py --dataset dataset.json --models model1,model2,model3 --limit 100
```

Options:
- `--dataset`: Path to the dataset file
- `--models`: Comma-separated list of models to evaluate
- `--output-dir`: Directory to save results (default: "results")
- `--limit`: number of functions to generate tests for

### Creating a Dataset

```bash
python collect_dataset.py --output data/dataset.json
```

Options:
- `--include-methods`: Include class methods in the dataset
- `--output`: Path to output file

### Fine-Tuning a Model

```bash
python fine_tune.py --model_name base_model_name --data_path dataset.json
```

Options:
- `--model_name`: Name or path of the base model to fine-tune
- `--data_path`: Path to the JSON dataset
- `--output_dir`: Directory to save the fine-tuned model (default: "./fine_tuned_model")
- `--num_epochs`: Number of training epochs (default: 3)

## Project Structure

- `generate_tests.py`: Main script for extracting code and generating tests
- `calculate_metrics.py`: Script for evaluating test quality metrics
- `collect_dataset.py`: Tool for creating datasets from existing code-test pairs
- `fine_tune.py`: Script for fine-tuning LLMs on test generation
- `test_metrics/`: Directory with evaluation metric implementations
- `data/`: Directory for datasets
- `requirements.txt`: Project dependencies
