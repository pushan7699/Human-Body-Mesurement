import os
import re


def fix_tf_imports(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check if file imports tensorflow
                if 'import tensorflow as tf' in content and 'tensorflow.compat.v1' not in content:
                    # Replace the import
                    new_content = content.replace(
                        'import tensorflow as tf',
                        'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()'
                    )

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Fixed: {filepath}")


# Run on src directory
fix_tf_imports('src')
print("Done!")