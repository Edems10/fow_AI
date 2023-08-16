import os

def extract_imports_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        imports = [line.strip() for line in lines if line.startswith(('import ', 'from '))]
    return imports

def find_imports_in_project(directory):
    all_imports = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                all_imports.extend(extract_imports_from_file(file_path))
    return all_imports

current_directory = os.getcwd()
imports = find_imports_in_project(current_directory)
unique_imports = list(set(imports))

for imp in unique_imports:
    print(imp)