import os

chumpy_file = os.path.join('.venv', 'lib', 'site-packages', 'chumpy', '__init__.py')

print(f"Fixing {chumpy_file}...")

if os.path.exists(chumpy_file):
    with open(chumpy_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Backup
    with open(chumpy_file + '.original', 'w', encoding='utf-8') as f:
        f.write(content)

    # The exact line in fresh chumpy
    old_line = 'from numpy import bool, int, float, complex, object, unicode, str, nan, inf'

    new_code = '''from numpy import nan, inf
import numpy as np
try:
    bool = np.bool_
    int = np.int_
    float = np.float_
    complex = np.complex_
    object = np.object_
    str = np.str_
    unicode = np.unicode_
except AttributeError:
    bool = bool
    int = int
    float = float
    complex = complex
    object = object
    str = str
    unicode = str'''

    if old_line in content:
        content = content.replace(old_line, new_code)

        with open(chumpy_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print("✓ Successfully patched chumpy!")
        print("Original saved as: " + chumpy_file + '.original')
    else:
        print("ERROR: Could not find the import line to replace")
        print("The file might already be modified or corrupted")
else:
    print(f"ERROR: File not found: {chumpy_file}")