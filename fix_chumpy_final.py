import os

chumpy_file = '.venv/lib/site-packages/chumpy/__init__.py'

if os.path.exists(chumpy_file):
    with open(chumpy_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace the problematic import line
    old_import = 'from numpy import bool, int, float, complex, object, unicode, str, nan, inf'

    new_import = '''from numpy import nan, inf
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

    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(chumpy_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✓ Successfully fixed chumpy!")
    else:
        print("Import line not found - file may already be modified")
else:
    print("Chumpy file not found")