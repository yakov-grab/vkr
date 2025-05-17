import os
import ast
import json
import shutil
import subprocess
from pathlib import Path
import argparse

def extract_functions(file_path, include_methods=True):
    """
    Extract all functions from a file
    
    Args:
        file_path: path to the file
        include_methods: whether to include class methods (default True)
    
    Returns:
        dict: dictionary with functions {function_name: function_code}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_method = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        if node in parent.body:
                            is_method = True
                            break
                
                if not include_methods and is_method:
                    continue
                
                func_name = node.name
                start_line = node.lineno - 1
                end_line = node.end_lineno
                func_code = '\n'.join(content.splitlines()[start_line:end_line])
                functions[func_name] = func_code
                
        return functions
    except SyntaxError:
        return {}

def extract_test_functions(file_path, include_methods=True):
    """
    Extract test functions from a file
    
    Args:
        file_path: path to the file
        include_methods: whether to include class methods (default True)
    
    Returns:
        dict: dictionary with test functions {function_name: function_code}
    """
    functions = extract_functions(file_path, include_methods)
    return {name: code for name, code in functions.items() if name.startswith('test_')}

def find_corresponding_implementation(test_name, functions_dict):
    """Find the corresponding implementation for a test"""
    if test_name.startswith('test_'):
        impl_name = test_name[5:]
        for func_name in functions_dict:
            if func_name == impl_name:
                return func_name
    return None

def collect_code_test_pairs(repo_path, include_methods=True):
    """
    Collect code-test pairs from a repository
    
    Args:
        repo_path: path to the repository
        include_methods: whether to include class methods (default True)
    
    Returns:
        list: list of pairs {function_code, test_code}
    """
    pairs = []
    implementation_files = []
    test_files = []
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if 'test_' in file or 'tests' in file_path.lower():
                    test_files.append(file_path)
                else:
                    implementation_files.append(file_path)
    
    all_functions = {}
    for file_path in implementation_files:
        functions = extract_functions(file_path, include_methods)
        all_functions.update(functions)
    
    for test_file in test_files:
        test_functions = extract_test_functions(test_file, include_methods)
        
        for test_name, test_code in test_functions.items():
            impl_name = find_corresponding_implementation(test_name, all_functions)
            if impl_name:
                pairs.append({
                    "function_code": all_functions[impl_name],
                    "test_code": test_code
                })
    
    return pairs

def clone_repository(repo_url, repo_path, branch=None):
    """
    Clone a repository using subprocess
    
    Args:
        repo_url: repository URL
        repo_path: path for cloning
        branch: branch to clone (default is main)
    
    Returns:
        bool: True if cloning is successful, False otherwise
    """
    if os.path.exists(repo_path):
        print(f"Directory {repo_path} already exists. Removing...")
        shutil.rmtree(repo_path)
    
    os.makedirs(os.path.dirname(repo_path) or '.', exist_ok=True)
    
    cmd = ["git", "clone", "--depth", "1"]
    
    if branch:
        cmd.extend(["--branch", branch])
    
    cmd.extend([repo_url, repo_path])
    
    print(f"Cloning repository {repo_url} to {repo_path}...")
    
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print("Cloning completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error when cloning the repository: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    repo_urls = [
        "https://github.com/psf/requests",
        "https://github.com/pallets/flask",
        "https://github.com/django/django",
        "https://github.com/scrapy/scrapy",
        "https://github.com/pandas-dev/pandas",
        "https://github.com/httpie/httpie",
        "https://github.com/pytest-dev/pytest",
        "https://github.com/numpy/numpy",
    ]
    repo_path = "./temp_repo"
    output_file = "python_code_test_dataset.json"
    
    parser = argparse.ArgumentParser(description='Collect code-test pairs from repositories')
    parser.add_argument('--include-methods', action='store_true', 
                      help='Include class methods in the dataset')
    parser.add_argument('--output', type=str, default=output_file,
                      help='Path to output file')
    args = parser.parse_args()
    
    all_pairs = []
    for repo_url in repo_urls:
        repo_path = f"./temp_repo/{repo_url.split('/')[-1]}"
        if clone_repository(repo_url, repo_path):
            pairs = collect_code_test_pairs(repo_path, args.include_methods)
            all_pairs.extend(pairs)
            print(f"Found {len(pairs)} code-test pairs in {repo_url}")
    
    with open(args.output, "w") as f:
        json.dump(all_pairs, f, indent=2)
    
    print(f"Dataset saved to file {args.output}")
    print(f"Total found {len(all_pairs)} code-test pairs")

if __name__ == "__main__":
    main()