import os
import ast
import json
import shutil
import subprocess
from pathlib import Path

def extract_functions(file_path):
    """Извлечение всех функций из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                start_line = node.lineno - 1
                end_line = node.end_lineno
                func_code = '\n'.join(content.splitlines()[start_line:end_line])
                functions[func_name] = func_code
                
        return functions
    except SyntaxError:
        return {}

def extract_test_functions(file_path):
    """Извлечение тестовых функций из файла"""
    functions = extract_functions(file_path)
    return {name: code for name, code in functions.items() if name.startswith('test_')}

def find_corresponding_implementation(test_name, functions_dict):
    """Поиск соответствующей реализации для теста"""
    if test_name.startswith('test_'):
        impl_name = test_name[5:]  # Убираем 'test_'
        for func_name in functions_dict:
            if func_name == impl_name:
                return func_name
    return None

def collect_code_test_pairs(repo_path):
    """Сбор пар код-тест из репозитория"""
    pairs = []
    implementation_files = []
    test_files = []
    
    # Находим все Python-файлы
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if 'test_' in file or 'tests' in file_path.lower():
                    test_files.append(file_path)
                else:
                    implementation_files.append(file_path)
    
    # Собираем все функции
    all_functions = {}
    for file_path in implementation_files:
        functions = extract_functions(file_path)
        all_functions.update(functions)
    
    # Собираем тесты и ищем соответствующие функции
    for test_file in test_files:
        test_functions = extract_test_functions(test_file)
        
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
    Клонирование репозитория с использованием subprocess
    
    Args:
        repo_url: URL репозитория
        repo_path: Путь для клонирования
        branch: Ветка для клонирования (по умолчанию основная)
    
    Returns:
        bool: True если клонирование успешно, False в противном случае
    """
    # Проверяем, существует ли уже директория
    if os.path.exists(repo_path):
        print(f"Директория {repo_path} уже существует. Удаляем...")
        shutil.rmtree(repo_path)
    
    # Создаем родительскую директорию, если она не существует
    os.makedirs(os.path.dirname(repo_path) or '.', exist_ok=True)
    
    # Формируем команду для git clone
    cmd = ["git", "clone", "--depth", "1"]
    
    # Добавляем ветку, если указана
    if branch:
        cmd.extend(["--branch", branch])
    
    # Добавляем URL и путь
    cmd.extend([repo_url, repo_path])
    
    print(f"Клонирование репозитория {repo_url} в {repo_path}...")
    
    try:
        # Выполняем команду клонирования
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print("Клонирование завершено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при клонировании репозитория: {e}")
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
    # repo_url = "https://github.com/psf/requests"
    repo_path = "./temp_repo"
    output_file = "python_code_test_dataset.json"
    
    for repo_url in repo_urls:
        repo_path = f"./temp_repo/{repo_url.split('/')[-1]}"
        if clone_repository(repo_url, repo_path):
            pairs = collect_code_test_pairs(repo_path)
        
        with open(output_file, "w") as f:
            json.dump(pairs, f, indent=2)
        
        print(f"Датасет сохранен в файл {output_file}")
        print(f"Найдено {len(pairs)} пар код-тест")
    else:
        print("Не удалось клонировать репозиторий. Выход.")

if __name__ == "__main__":
    main()