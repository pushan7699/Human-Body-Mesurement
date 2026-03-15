import os
import re

def find_contrib_imports(directory):
    issues = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines, 1):
                        if 'tensorflow.contrib' in line or 'tf.contrib' in line:
                            issues.append(f"{filepath}:{i} -> {line.strip()}")
    return issues

print("Finding all tensorflow.contrib usage...")
issues = find_contrib_imports('src')
if issues:
    for issue in issues:
        print(issue)
else:
    print("No contrib imports found!")