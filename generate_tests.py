import os
import ast
import re
import argparse
from typing import List, Dict, Set, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class FunctionExtractor:
    """Extract functions from Python source code."""

    @staticmethod
    def extract_functions(source_code: str, source_file: str = "example.py") -> List[Dict]:
        """
        Extract all function definitions from the provided source code.
        
        Args:
            source_code: String containing Python source code
            source_file: Name of the source file (used for module name)
            
        Returns:
            List of dictionaries containing function information
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"Error parsing source code: {e}")
            return []
        
        functions = []
        module_name = os.path.splitext(os.path.basename(source_file))[0]
        
        all_imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines = source_code.splitlines()[node.lineno-1:node.end_lineno]
                all_imports.append('\n'.join(import_lines))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                
                function_lines = source_code.splitlines()[node.lineno-1:node.end_lineno]
                function_code = '\n'.join(function_lines)
                
                function_ast = ast.parse(function_code)
                dependencies = FunctionExtractor._find_dependencies(function_ast, tree, source_code)
                
                args = []
                for arg in node.args.args:
                    arg_name = arg.arg
                    arg_type = None
                    if arg.annotation:
                        arg_type = source_code.splitlines()[arg.annotation.lineno-1][arg.annotation.col_offset:arg.annotation.end_col_offset]
                    args.append((arg_name, arg_type))
                
                return_type = None
                if node.returns:
                    return_type = source_code.splitlines()[node.returns.lineno-1][node.returns.col_offset:node.returns.end_col_offset]
                
                parent_class = None
                class_code = None
                for potential_parent in ast.walk(tree):
                    if isinstance(potential_parent, ast.ClassDef):
                        for child in ast.iter_child_nodes(potential_parent):
                            if isinstance(child, ast.FunctionDef) and child.name == node.name:
                                parent_class = potential_parent.name
                                class_lines = source_code.splitlines()[potential_parent.lineno-1:potential_parent.end_lineno]
                                class_code = '\n'.join(class_lines)
                                break
                
                functions.append({
                    'name': node.name,
                    'code': function_code,
                    'class_code': class_code,
                    'parent_class': parent_class,
                    'docstring': docstring,
                    'args': args,
                    'return_type': return_type,
                    'imports': all_imports,
                    'dependencies': dependencies,
                    'module_name': module_name
                })
            elif isinstance(node, ast.ClassDef):
                for method_node in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                    docstring = ast.get_docstring(method_node)
                    
                    method_lines = source_code.splitlines()[method_node.lineno-1:method_node.end_lineno]
                    method_code = '\n'.join(method_lines)
                    
                    class_lines = source_code.splitlines()[node.lineno-1:node.end_lineno]
                    class_code = '\n'.join(class_lines)
                    
                    method_ast = ast.parse(method_code)
                    dependencies = FunctionExtractor._find_dependencies(method_ast, tree, source_code)
                    
                    args = []
                    for arg in method_node.args.args:
                        if arg.arg == 'self':
                            continue
                        arg_name = arg.arg
                        arg_type = None
                        if arg.annotation:
                            arg_type = source_code.splitlines()[arg.annotation.lineno-1][arg.annotation.col_offset:arg.annotation.end_col_offset]
                        args.append((arg_name, arg_type))
                    
                    return_type = None
                    if method_node.returns:
                        return_type = source_code.splitlines()[method_node.returns.lineno-1][method_node.returns.col_offset:method_node.returns.end_col_offset]
                    
                    functions.append({
                        'name': method_node.name,
                        'code': method_code,
                        'class_code': class_code,
                        'parent_class': node.name,
                        'docstring': docstring,
                        'args': args,
                        'return_type': return_type,
                        'imports': all_imports,
                        'dependencies': dependencies,
                        'module_name': module_name
                    })
        
        return functions
    
    @staticmethod
    def _find_dependencies(node_ast: ast.AST, full_tree: ast.AST, source_code: str) -> Set[str]:
        """
        Find dependencies of a function or method.
        
        Args:
            node_ast: AST of the function or method
            full_tree: AST of the entire source file
            source_code: Full source code
            
        Returns:
            Set of dependency names
        """
        dependencies = set()
        
        for node in ast.walk(node_ast):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                dependencies.add(node.id)
        
        return dependencies

class TestGenerator:
    """Generate pytest unit tests using a local LLM."""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"):
        """
        Initialize the test generator.
        
        Args:
            model_name: Name of the local LLM to use
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            print(f"Loading model {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
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
            if function_info.get('parent_class'):
                class_info = f"\nThis function is a method of the class '{function_info['parent_class']}':\n\n```python\n{function_info['class_code']}\n```\n"
                prompt = prompt.replace("unit test for", f"unit test for a method of class '{function_info['parent_class']}' for")
            
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
            return f"# Error generating test: {e}"
    
    def _clean_test_code(self, test_code: str, function_info: Dict) -> str:
        """
        Clean up the generated test code.
        
        Args:
            test_code: Generated test code
            function_info: Dictionary containing function information
            
        Returns:
            Cleaned test code
        """
        lines = test_code.split('\n')
        cleaned_lines = []
        imported_modules = set()
        has_fixture = '@fixture' in test_code
        has_pytest_import = False
        has_fixture_import = False
        
        for line in lines:
            # Проверяем импорты pytest
            if line.startswith('import pytest'):
                has_pytest_import = True
                cleaned_lines.append(line)
                continue
                
            if line.startswith('from pytest import') and 'fixture' in line:
                has_fixture_import = True
                cleaned_lines.append(line)
                continue
                
            import_match = re.match(r'^from\s+(\w+)(\.\w+)*\s+import', line)
            if import_match:
                module = import_match.group(1)
                if module == function_info['module_name']:
                    continue
                if module in imported_modules:
                    continue
                imported_modules.add(module)
                
            cleaned_lines.append(line)
        
        # Добавляем необходимые импорты если их нет
        if has_fixture and not has_pytest_import and not has_fixture_import:
            cleaned_lines.insert(0, 'import pytest')
            
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    def _create_prompt(self, function_info: Dict) -> str:
        """
        Create a prompt for the model to generate a test.
        
        Args:
            function_info: Dictionary containing function information
            
        Returns:
            String containing the prompt
        """
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
- If you use pytest features like @fixture, make sure to include 'import pytest' at the beginning
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
        """
        Extract test code from generated text.
        
        Args:
            generated_text: Generated text from the model
            
        Returns:
            String containing the test code
        """
        test_pattern = r"```python\s*(.*?)```"
        match = re.search(test_pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        match = re.search(r"((?:import\s+.*?$)|(?:def\s+test_.*?))", generated_text, re.DOTALL | re.MULTILINE)
        if match:
            start_idx = match.start()
            return generated_text[start_idx:].strip()
        
        return generated_text.strip()

def analyze_imports(functions: List[Dict]) -> Dict[str, Set[str]]:
    """
    Analyze imports needed for all functions.
    
    Args:
        functions: List of function information dictionaries
        
    Returns:
        Dictionary mapping module names to sets of required imports
    """
    module_imports = {}
    
    for function_info in functions:
        module_name = function_info.get('module_name')
        if module_name not in module_imports:
            module_imports[module_name] = set()
            
        for import_line in function_info.get('imports', []):
            module_imports[module_name].add(import_line)
    
    return module_imports

def generate_test_file(functions: List[Dict], test_generator: TestGenerator, output_dir: str, source_file: str) -> None:
    """
    Generate a test file for the given functions.
    
    Args:
        functions: List of function information dictionaries
        test_generator: TestGenerator instance
        output_dir: Directory to save the test file
        source_file: Source file path
    """
    module_name = os.path.splitext(os.path.basename(source_file))[0]
    test_file_path = os.path.join(output_dir, f"test_{module_name}.py")
    
    module_imports = analyze_imports(functions)
    required_imports = module_imports.get(module_name, set())
    
    class_methods = {}
    standalone_functions = []
    
    for function_info in functions:
        if function_info.get('parent_class'):
            parent_class = function_info['parent_class']
            if parent_class not in class_methods:
                class_methods[parent_class] = []
            class_methods[parent_class].append(function_info)
        else:
            standalone_functions.append(function_info)
    
    with open(test_file_path, 'w') as f:
        f.write(f"# Generated pytest unit tests for {module_name}.py\n")
        
        f.write("import pytest\n")
        f.write(f"from {module_name} import *\n")
        
        f.write("""
# Standard library imports
import os
import sys
import io
import re
from unittest.mock import patch, MagicMock, Mock

# Common test imports
import tempfile
from contextlib import contextmanager
from pathlib import Path

""")
        
        for function_info in standalone_functions:
            print(f"Generating test for function '{function_info['name']}'...")
            test_code = test_generator.generate_test(function_info)
            f.write(f"\n# Test for function '{function_info['name']}'\n")
            f.write(test_code)
            f.write("\n\n")
        
        for class_name, methods in class_methods.items():
            f.write(f"\n# Tests for class '{class_name}'\n")
            f.write(f"class Test{class_name}:\n")
            
            f.write("    @pytest.fixture\n")
            f.write(f"    def {class_name.lower()}_instance(self):\n")
            f.write(f"        # Setup code - create an instance of {class_name}\n")
            f.write(f"        instance = {class_name}()\n")
            f.write(f"        yield instance\n")
            f.write(f"        # Teardown code if needed\n\n")
            
            for method_info in methods:
                print(f"Generating test for method '{class_name}.{method_info['name']}'...")
                test_code = test_generator.generate_test(method_info)
                
                indented_test_code = "    " + test_code.replace("\n", "\n    ")
                
                f.write(f"    # Test for method '{method_info['name']}'\n")
                f.write(indented_test_code)
                f.write("\n\n")
    
    print(f"Tests generated and saved to {test_file_path}")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Extract functions from Python code and generate pytest unit tests using a local LLM")
    parser.add_argument("source_file", help="Path to the Python source file")
    parser.add_argument("--output-dir", default="tests", help="Directory to save the generated tests")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct", help="Name of the local LLM to use")
    args = parser.parse_args()
    
    if not os.path.exists(args.source_file):
        print(f"Error: Source file '{args.source_file}' not found.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.source_file, 'r') as f:
        source_code = f.read()
    
    functions = FunctionExtractor.extract_functions(source_code, args.source_file)
    
    if not functions:
        print("No function definitions found in the source file.")
        return
    
    print(f"Found {len(functions)} function(s) in the source file.")
    
    test_generator = TestGenerator(model_name=args.model)
    
    generate_test_file(functions, test_generator, args.output_dir, args.source_file)

if __name__ == "__main__":
    main() 