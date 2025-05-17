#!/usr/bin/env python3
"""
Metrics collector for LLM-generated Python tests

This script automates:
1. Test generation using different LLM models
2. Test execution and result collection
3. Metrics calculation (CodeBLEU, Pass@k, Syntax Validity)
4. Comparison of models' performance

Usage:
  python metrics_collector.py --dataset test_dataset.json --models model1,model2,model3
"""

import os
import sys
import json
import argparse
import tempfile
import ast
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
import pytest
from tqdm import tqdm
from codebleu import calc_codebleu
import coverage

sys.path.append('.')
from generate_tests import TestGenerator, FunctionExtractor


class SyntaxValidator:
    """Validate Python code syntax"""
    
    @staticmethod
    def is_valid_python(code: str) -> bool:
        """Check if the provided code is valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        
    @staticmethod
    def validate_tests(test_code: str) -> Dict[str, Any]:
        """Validate the test code and return validation metrics"""
        result = {
            "is_valid": False,
            "error_message": "",
            "test_count": 0,
            "has_imports": False,
            "has_pytest_imports": False,
        }
        
        try:
            tree = ast.parse(test_code)
            result["is_valid"] = True
            
            test_count = 0
            has_pytest_imports = False
            has_imports = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_count += 1
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    has_imports = True
                    if "pytest" in ast.unparse(node):
                        has_pytest_imports = True
            
            result["test_count"] = test_count
            result["has_imports"] = has_imports
            result["has_pytest_imports"] = has_pytest_imports
            
        except SyntaxError as e:
            result["is_valid"] = False
            result["error_message"] = str(e)
        
        return result


class TestExecutor:
    """Execute generated tests and collect results"""
    
    @staticmethod
    def execute_test(test_file: str, source_file: str) -> Dict[str, Any]:
        """Execute tests and return results"""
        result = {
            "success": False,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "execution_time": 0,
            "error_message": "",
            "test_details": [],
            "coverage": {
                "line_rate": 0.0,
                "branch_rate": 0.0,
                "covered_lines": 0,
                "total_lines": 0
            }
        }
        
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                conftest_path = Path(tempdir) / "conftest.py"
                with open(conftest_path, 'w') as f:
                    f.write("import pytest\n")
                
                sys.path.insert(0, str(Path(source_file).parent.absolute()))
                
                # Add the proper import for pytest fixture
                with open(test_file, 'r') as f:
                    test_content = f.read()
                
                if '@fixture' in test_content and 'from pytest import fixture' not in test_content and 'import pytest' not in test_content:
                    # Add pytest import at the beginning of the file
                    with open(test_file, 'w') as f:
                        f.write("import pytest\n" + test_content)
                
                items = pytest.main(["--collect-only", test_file, "-v"])
                if items != 0:
                    result["error_message"] = "Failed to collect tests"
                    return result
                
                pytest_results = []
                
                class ResultCollector:
                    @staticmethod
                    def pytest_runtest_logreport(report):
                        if report.when == 'call' or (report.when == 'setup' and report.outcome != 'passed'):
                            pytest_results.append({
                                'name': report.nodeid,
                                'outcome': report.outcome,
                                'duration': getattr(report, 'duration', 0),
                                'message': str(getattr(report, 'longrepr', ''))
                            })
                
                # Setup coverage
                source_module = Path(source_file).stem
                
                cov = coverage.Coverage(
                    source=[source_module],
                    data_file=str(Path(tempdir) / ".coverage"),
                    branch=True
                )
                
                try:
                    cov.start()
                    exit_code = pytest.main([test_file, "-v"], plugins=[ResultCollector()])
                    cov.stop()
                    cov.save()
                
                    # Get coverage data
                    if os.path.exists(str(Path(tempdir) / ".coverage")):
                        cov.load()
                        
                        # Get file data
                        data = cov.get_data()
                        if data and source_module in data.measured_files():
                            file_data = list(data.measured_files())[0]  # Get the first file
                            
                            # Get analysis
                            analysis = cov.analysis2(file_data)
                            if analysis:
                                statements = analysis[0]  # executable lines
                                missing = analysis[1]     # missing lines
                                branches = analysis[2]    # branches
                                missing_branches = analysis[3]  # missing branches
                                
                                if statements:
                                    covered_lines = len(statements) - len(missing)
                                    line_rate = covered_lines / len(statements) if statements else 0
                                    
                                    # Calculate branch coverage if available
                                    branch_rate = 0.0
                                    if branches:
                                        covered_branches = len(branches) - len(missing_branches) 
                                        branch_rate = covered_branches / len(branches) if len(branches) > 0 else 0.0
                                    
                                    result["coverage"] = {
                                        "line_rate": line_rate,
                                        "branch_rate": branch_rate,
                                        "covered_lines": covered_lines,
                                        "total_lines": len(statements),
                                        "branch_covered": len(branches) - len(missing_branches) if branches else 0,
                                        "branch_total": len(branches) if branches else 0
                                    }
                except Exception as e:
                    print(f"Error during coverage collection: {e}")
                
                result["success"] = exit_code == 0
                result["total"] = len(pytest_results)
                result["passed"] = sum(1 for r in pytest_results if r['outcome'] == 'passed')
                result["failed"] = sum(1 for r in pytest_results if r['outcome'] == 'failed')
                result["error"] = sum(1 for r in pytest_results if r['outcome'] == 'error')
                result["skipped"] = sum(1 for r in pytest_results if r['outcome'] == 'skipped')
                result["execution_time"] = sum(r['duration'] for r in pytest_results)
                result["test_details"] = pytest_results
                
        except Exception as e:
            result["error_message"] = str(e)
        finally:
            if str(Path(source_file).parent.absolute()) in sys.path:
                sys.path.remove(str(Path(source_file).parent.absolute()))
        
        return result


class CodeBLEUCalculator:
    """Calculate CodeBLEU metrics between generated and reference code"""
    
    @staticmethod
    def calculate_code_bleu(generated_code: str, reference_code: str) -> float:
        """
        Calculate CodeBLEU score between generated and reference code.
        
        CodeBLEU is a weighted combination of:
        1. n-gram match (BLEU)
        2. weighted n-gram match (BLEU-weighted)
        3. AST match
        4. data-flow match
        """
        if not generated_code or not reference_code:
            print("Warning: Empty code provided for CodeBLEU calculation")
            return 0.0
        
        result = calc_codebleu(
            [reference_code],
            [generated_code],
            lang="python",
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )
        
        return result['codebleu']


class LocalModelTestGenerator:
    """Test generator for local model files"""
    
    def __init__(self, model_path: str, cpu_only: bool = False, cuda_blocking: bool = False):
        """
        Initialize the test generator with a local model file.
        
        Args:
            model_path: Path to the local model file or directory
            cpu_only: Force CPU-only mode even if CUDA is available
            cuda_blocking: Enable CUDA_LAUNCH_BLOCKING for better error messages
        """
        self.model_path = model_path
        print(f"Using local model from: {model_path}")
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            if cuda_blocking:
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                print("Enabled CUDA_LAUNCH_BLOCKING=1 for synchronous CUDA operations")
            
            if cpu_only:
                self.device = "cpu"
                print("Forcing CPU-only mode as requested")
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {self.device}")
            
            model_dir = model_path
            if os.path.isfile(model_path):
                model_dir = os.path.dirname(model_path)
                if not model_dir:
                    model_dir = "."
            
            print(f"Loading tokenizer from {model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            print(f"Loading model from {model_dir}...")
            
            if self.device == "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            print("Model loaded successfully")
            
            from transformers import pipeline
            
            try:
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer
                )
            except Exception as e:
                print(f"Error creating pipeline with standard approach: {e}")
                print("Trying simplified fallback approach...")
                if self.device == "cpu":
                    print("Loading model directly without accelerate (CPU mode)")
                    self.model = self.model.to("cpu")
                    self.generator = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device="cpu"
                    )
                else:
                    print("Using basic pipeline initialization")
                    self.generator = pipeline(
                        "text-generation",
                        model=self.model.name_or_path if hasattr(self.model, "name_or_path") else model_dir,
                        tokenizer=self.tokenizer
                    )
                print("Fallback approach succeeded")
            
        except Exception as e:
            print(f"Error loading local model: {e}")
            raise
    
    def generate_test(self, function_info: Dict) -> str:
        """
        Generate a pytest unit test for the given function.
        
        Args:
            function_info: Dictionary containing function information
            
        Returns:
            String containing the pytest unit test
        """
        prompt = self._create_prompt(function_info)
        
        try:
            response = self.generator(
                prompt,
                max_length=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1
            )
            
            test_code = response[0]['generated_text']
            
            test_code = test_code[len(prompt):].strip()
            
            test_code = self._extract_test_code(test_code)
            
            test_code = self._clean_test_code(test_code, function_info)
            
            return test_code
        except Exception as e:
            print(f"Error generating test: {e}")
            
            if "CUDA" in str(e):
                print("This appears to be a CUDA error. Consider:")
                print("1. Running with --cpu-only to use CPU instead of GPU")
                print("2. Using --cuda-debugging for better error traces")
                print("3. Reducing batch size or model complexity")
            
            return f"# Error generating test: {e}"
    
    def _create_prompt(self, function_info: Dict) -> str:
        """Same as in the original TestGenerator class"""
        module_name = function_info.get('module_name', 'example')
        imports = '\n'.join(function_info.get('imports', [])) if function_info.get('imports') else ''
        
        docstring_info = ""
        if function_info.get('docstring'):
            docstring_info = f"\nFunction docstring:\n'''\n{function_info['docstring']}\n'''"
        
        type_info = ""
        if function_info.get('args') or function_info.get('return_type'):
            type_info = "\nSignature information:"
            if function_info.get('args'):
                type_info += "\nArguments:"
                for arg_name, arg_type in function_info['args']:
                    type_info += f"\n  - {arg_name}"
                    if arg_type:
                        type_info += f": {arg_type}"
            if function_info.get('return_type'):
                type_info += f"\nReturn type: {function_info['return_type']}"
        
        prompt = f"""
Write a pytest unit test for the following Python function from the module '{module_name}':

```python
{function_info['code']}
```
{docstring_info}
{type_info}

Imports used in the module:
```python
{imports}
```

Write a comprehensive pytest test for this function. The function is already imported as 'from {module_name} import *'.

Your test should:
1. Include both normal cases and edge cases
2. Use appropriate fixtures to set up preconditions
3. Test expected behavior and edge cases
4. Use clear and descriptive assertion messages
5. Follow pytest best practices
6. Be concise and focused

Important guidelines:
- DO NOT import the function being tested, it's already imported from '{module_name}'
- Include necessary imports for any external libraries used in the test (but not pytest)
- Use parametrize for testing multiple input/output combinations
- Use fixtures appropriately for setup/teardown
- Include proper mocking for external dependencies if needed
- Add descriptive docstrings to test functions

Generate ONLY the test code, no explanations or comments outside the code.
"""
        
        if function_info.get('parent_class'):
            prompt += f"""
Since this is a method of class '{function_info['parent_class']}', your test should:
1. Create a fixture for the class instance
2. Test the method in the context of its class
3. Consider class state and any side effects

Class definition:
```python
{function_info['class_code']}
```
"""
        
        return prompt
    
    def _extract_test_code(self, generated_text: str) -> str:
        """Same as in the original TestGenerator class"""
        test_pattern = r"```python\s*(.*?)```"
        match = re.search(test_pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        match = re.search(r"((?:import\s+.*?$)|(?:def\s+test_.*?))", generated_text, re.DOTALL | re.MULTILINE)
        if match:
            start_idx = match.start()
            return generated_text[start_idx:].strip()
        
        return generated_text.strip()
    
    def _clean_test_code(self, test_code: str, function_info: Dict) -> str:
        """Same as in the original TestGenerator class"""
        lines = test_code.split('\n')
        cleaned_lines = []
        imported_modules = set()
        
        for line in lines:
            if line.startswith('import pytest') or line.startswith('from pytest'):
                continue
                
            import_match = re.match(r'^from\s+(\w+)(\.\w+)*\s+import', line)
            if import_match:
                module = import_match.group(1)
                if module == function_info['module_name'] or module == 'pytest':
                    continue
                if module in imported_modules:
                    continue
                imported_modules.add(module)
                
            cleaned_lines.append(line)
        
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)


class MetricsCollector:
    """Main class to collect and calculate metrics for generated tests"""
    
    def __init__(self, models: List[str], dataset_path: str, output_dir: str = "results"):
        self.models = models
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.cpu_only = False
        self.cuda_debugging = False
        
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        self.model_dirs = {}
        for model in models:
            dir_name = model.replace('/', '_').replace('\\', '_')
            self.model_dirs[model] = dir_name
            
            model_dir = self.output_dir / dir_name
            model_dir.mkdir(exist_ok=True, parents=True)
            (model_dir / "tests").mkdir(exist_ok=True)
            (model_dir / "results").mkdir(exist_ok=True)
    
    def generate_tests_for_model(self, model_name: str) -> Dict[str, Any]:
        """Generate tests using specified model and return results"""
        print(f"Generating tests using model: {model_name}")
        
        model_results = {
            "model": model_name,
            "total_functions": len(self.dataset),
            "generated_tests": 0,
            "valid_syntax": 0,
            "total_generation_time": 0,
            "total_execution_time": 0,
            "errors": [],
            "items": []
        }
        
        try:
            if os.path.exists(model_name) or model_name.startswith('./') or model_name.startswith('../'):
                test_generator = LocalModelTestGenerator(model_name, self.cpu_only, self.cuda_debugging)
            else:
                test_generator = TestGenerator(model_name=model_name)
        except Exception as e:
            print(f"Error initializing model {model_name}: {e}")
            model_results["errors"].append({
                "type": "model_initialization",
                "message": str(e)
            })
            return model_results
        
        dir_name = self.model_dirs[model_name]
        
        for idx, item in enumerate(tqdm(self.dataset, desc=f"Generating tests for {model_name}")):
            try:
                function_code = item.get("function_code", "")
                reference_test = item.get("test_code", "")
                
                if not function_code or not reference_test:
                    print(f"Warning: Empty function or test code for item {idx}")
                    continue
                
                temp_func_file = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
                temp_func_filename = temp_func_file.name
                
                with open(temp_func_filename, 'w') as f:
                    f.write(function_code)
                
                functions = FunctionExtractor.extract_functions(function_code, temp_func_filename)
                
                if not functions:
                    print(f"Warning: No functions extracted from item {idx}")
                    continue
                
                function_info = functions[0]
                
                generation_start = time.time()
                try:
                    generated_test = test_generator.generate_test(function_info)
                except Exception as e:
                    print(f"Error generating test for item {idx}: {e}")
                    model_results["errors"].append({
                        "item_id": idx,
                        "type": "test_generation",
                        "message": str(e)
                    })
                    continue
                
                generation_time = time.time() - generation_start
                model_results["total_generation_time"] += generation_time
                
                syntax_validation = SyntaxValidator.validate_tests(generated_test)
                
                temp_test_file = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
                temp_test_filename = temp_test_file.name
                
                with open(temp_test_filename, 'w') as f:
                    f.write("import sys\nimport os\nimport pytest\n")
                    f.write(f"sys.path.insert(0, os.path.dirname('{temp_func_filename}'))\n")
                    f.write(f"from {Path(temp_func_filename).stem} import *\n\n")
                    f.write(generated_test)
                
                item_filename = f"func_{idx}.py"
                test_filename = f"test_func_{idx}.py"
                
                with open(self.output_dir / dir_name / "tests" / item_filename, 'w') as f:
                    f.write(function_code)
                
                with open(self.output_dir / dir_name / "tests" / test_filename, 'w') as f:
                    f.write(generated_test)
                
                execution_result = {}
                if syntax_validation["is_valid"]:
                    try:
                        execution_result = TestExecutor.execute_test(temp_test_filename, temp_func_filename)
                        model_results["total_execution_time"] += execution_result.get("execution_time", 0)
                    except Exception as e:
                        print(f"Error executing test for item {idx}: {e}")
                        model_results["errors"].append({
                            "item_id": idx,
                            "type": "test_execution",
                            "message": str(e)
                        })
                        execution_result = {
                            "success": False,
                            "error_message": str(e)
                        }
                
                try:
                    code_bleu = CodeBLEUCalculator.calculate_code_bleu(generated_test, reference_test)
                except Exception as e:
                    print(f"Error calculating CodeBLEU for item {idx}: {e}")
                    code_bleu = 0.0
                
                item_result = {
                    "id": idx,
                    "function_code": function_code,
                    "reference_test": reference_test,
                    "generated_test": generated_test,
                    "syntax_validation": syntax_validation,
                    "execution_result": execution_result,
                    "code_bleu": code_bleu,
                    "generation_time": generation_time
                }
                
                model_results["items"].append(item_result)
                
                if syntax_validation["is_valid"]:
                    model_results["valid_syntax"] += 1
                
                model_results["generated_tests"] += 1
                
                os.unlink(temp_func_filename)
                os.unlink(temp_test_filename)
                
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                model_results["errors"].append({
                    "item_id": idx,
                    "type": "general",
                    "message": str(e)
                })
                continue
        
        if model_results["generated_tests"] > 0:
            model_results["average_generation_time"] = model_results["total_generation_time"] / model_results["generated_tests"]
            model_results["average_execution_time"] = model_results["total_execution_time"] / model_results["generated_tests"]
        
        model_results["error_stats"] = {
            "total_errors": len(model_results["errors"]),
            "error_types": {
                error["type"]: sum(1 for e in model_results["errors"] if e["type"] == error["type"])
                for error in model_results["errors"]
            }
        }
        
        with open(self.output_dir / dir_name / "results.json", 'w') as f:
            json.dump(model_results, f, indent=2)
        
        return model_results
    
    def calculate_pass_at_k(self, model_results: Dict[str, Any], k: int = 1) -> float:
        """
        Calculate Pass@k metric
        
        Pass@k measures what fraction of problems are solved if allowed k sample attempts
        """
        items = model_results["items"]
        passes = [item["execution_result"].get("success", False) for item in items]
        
        if not passes:
            return 0.0
        
        return sum(passes) / len(passes)
    
    def calculate_metrics(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregated metrics from model results"""
        items = model_results["items"]
        
        metrics = {
            "model": model_results["model"],
            "total_functions": model_results["total_functions"],
            "generated_tests": model_results["generated_tests"],
            "valid_syntax_percent": model_results["valid_syntax"] / model_results["generated_tests"] * 100 if model_results["generated_tests"] > 0 else 0,
            "syntax_validity": model_results["valid_syntax"] / model_results["generated_tests"] if model_results["generated_tests"] > 0 else 0,
            "pass@1": self.calculate_pass_at_k(model_results, k=1),
            "average_code_bleu": np.mean([item["code_bleu"] for item in items]) if items else 0,
            "test_execution": {
                "passed_percent": np.mean([item["execution_result"].get("passed", 0) / max(1, item["execution_result"].get("total", 1)) * 100 for item in items if item["execution_result"].get("total", 0) > 0]) if items else 0,
                "failed_percent": np.mean([item["execution_result"].get("failed", 0) / max(1, item["execution_result"].get("total", 1)) * 100 for item in items if item["execution_result"].get("total", 0) > 0]) if items else 0,
                "error_percent": np.mean([item["execution_result"].get("error", 0) / max(1, item["execution_result"].get("total", 1)) * 100 for item in items if item["execution_result"].get("total", 0) > 0]) if items else 0,
            },
            "timing": {
                "total_generation_time": model_results.get("total_generation_time", 0),
                "total_execution_time": model_results.get("total_execution_time", 0),
                "average_generation_time": model_results.get("average_generation_time", 0),
                "average_execution_time": model_results.get("average_execution_time", 0)
            }
        }
        
        # Calculate average code coverage
        coverage_values = []
        for item in items:
            if "execution_result" in item and "coverage" in item["execution_result"]:
                cov = item["execution_result"]["coverage"]
                if cov["total_lines"] > 0:
                    coverage_values.append(cov["line_rate"] * 100)
                    
        if coverage_values:
            metrics["coverage_percent"] = np.mean(coverage_values)
        else:
            metrics["coverage_percent"] = 0.0
        
        return metrics
    
    def run_all(self) -> Dict[str, Any]:
        """Run the entire metrics collection pipeline for all models"""
        all_results = {
            "dataset": self.dataset_path,
            "total_functions": len(self.dataset),
            "models": []
        }
        
        for model in self.models:
            model_results = self.generate_tests_for_model(model)
            model_metrics = self.calculate_metrics(model_results)
            all_results["models"].append(model_metrics)
        
        with open(self.output_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def generate_comparison_report(self, all_results: Dict[str, Any]) -> str:
        """Generate a comparison report in markdown format"""
        md_report = f"# LLM Test Generation Comparison Report\n\n"
        md_report += f"## Dataset: {Path(self.dataset_path).name}\n\n"
        md_report += f"- Total functions analyzed: {all_results['total_functions']}\n\n"
        
        md_report += "## Model Comparison\n\n"
        
        md_report += "| Model | Total Generated Tests | Skipped Functions | Valid Syntax (%) | Pass@1 | Avg. CodeBLEU | Tests Passed (%) | Tests Failed (%) | Coverage (%) | Avg. Gen Time (s) | Avg. Exec Time (s) |\n"
        md_report += "|-------|-----------------------|-------------------|------------------|--------|---------------|------------------|------------------|--------------|-------------------|--------------------|\n"
        
        for model_metric in all_results["models"]:
            skipped = all_results['total_functions'] - model_metric['generated_tests']
            md_report += f"| {model_metric['model']} | "
            md_report += f"{model_metric['generated_tests']} | "
            md_report += f"{skipped} | "
            md_report += f"{model_metric['valid_syntax_percent']:.2f} | "
            md_report += f"{model_metric['pass@1']:.2f} | "
            md_report += f"{model_metric['average_code_bleu']:.2f} | "
            md_report += f"{model_metric['test_execution']['passed_percent']:.2f} | "
            md_report += f"{model_metric['test_execution']['failed_percent']:.2f} | "
            coverage_percent = model_metric.get('coverage_percent', 0.0)
            md_report += f"{coverage_percent:.2f} | "
            md_report += f"{model_metric['timing']['average_generation_time']:.2f} | "
            md_report += f"{model_metric['timing']['average_execution_time']:.2f} |\n"
        
        report_path = self.output_dir / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(md_report)
        
        return md_report


def main():
    parser = argparse.ArgumentParser(description="Collect metrics for LLM-generated tests")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of model names")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of functions to process (for testing)")
    parser.add_argument("--analyze-only", action="store_true", help="Skip test generation and only analyze existing results")
    parser.add_argument("--local-model-dir", type=str, default=None, help="Directory containing local model files (overrides model names for local models)")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode for model inference (helps with CUDA errors)")
    parser.add_argument("--cuda-debugging", action="store_true", help="Enable CUDA_LAUNCH_BLOCKING=1 for better error messages")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip coverage measurement to improve performance")
    
    args = parser.parse_args()
    
    models = [model.strip() for model in args.models.split(",")]
    
    if args.local_model_dir:
        for i, model in enumerate(models):
            if not model.startswith("http") and not os.path.exists(model):
                local_path = os.path.join(args.local_model_dir, model)
                if os.path.exists(local_path):
                    models[i] = local_path
                    print(f"Using local model at: {local_path}")
    
    print(f"Running metrics collection for models: {', '.join(models)}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output}")
    if args.limit:
        print(f"Processing only first {args.limit} functions")
    if args.cpu_only:
        print("Using CPU-only mode for model inference")
    if args.cuda_debugging:
        print("CUDA debugging enabled with CUDA_LAUNCH_BLOCKING=1")
    
    collector = MetricsCollector(models, args.dataset, args.output)
    
    collector.cpu_only = args.cpu_only
    collector.cuda_debugging = args.cuda_debugging
    
    if args.limit and args.limit > 0:
        collector.dataset = collector.dataset[:args.limit]
    
    all_results = {"dataset": args.dataset, "total_functions": len(collector.dataset), "models": []}
    
    if args.analyze_only:
        print("Skipping test generation, analyzing existing results...")
        for model in models:
            dir_name = model.replace('/', '_').replace('\\', '_')
            result_path = collector.output_dir / dir_name / "results.json"
            
            if result_path.exists():
                print(f"Loading results for {model}...")
                with open(result_path, 'r') as f:
                    model_results = json.load(f)
                    model_metrics = collector.calculate_metrics(model_results)
                    all_results["models"].append(model_metrics)
            else:
                print(f"No results found for {model} at {result_path}")
    else:
        all_results = collector.run_all()
    
    with open(collector.output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nGenerated comparison report:")
    report = collector.generate_comparison_report(all_results)
    print(f"\nReport saved to {args.output}/comparison_report.md")


if __name__ == "__main__":
    main() 