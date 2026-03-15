import os

chumpy_init = r'.venv\lib\site-packages\chumpy\__init__.py'

print(f"Looking for chumpy at: {chumpy_init}")

if os.path.exists(chumpy_init):
    print("Found chumpy, patching...")

    with open(chumpy_init, 'r') as f:
        content = f.read()

    # Backup
    with open(chumpy_init + '.backup', 'w') as f:
        f.write(content)
    print("Backup created")

    old_line = 'from numpy import bool, int, float, complex, object, unicode, str, nan, inf'
    new_lines = '''from numpy import complex, unicode, nan, inf
import numpy as np
bool = np.bool_
int = np.int_
float = np.float_
object = np.object_
str = np.str_'''

    if old_line in content:
        content = content.replace(old_line, new_lines)
        with open(chumpy_init, 'w') as f:
            f.write(content)
        print("✓ Patched chumpy successfully!")
    else:
        print("Line not found - maybe already patched?")
else:
    print(f"ERROR: Chumpy not found at {chumpy_init}")
    print("Searching for chumpy...")

    # Search for it
    import site

    for path in site.getsitepackages():
        chumpy_path = os.path.join(path, 'chumpy', '__init__.py')
        if os.path.exists(chumpy_path):
            print(f"Found at: {chumpy_path}")