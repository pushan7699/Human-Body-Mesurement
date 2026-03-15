filepath = 'demo.py'

print(f"Fixing {filepath}...")

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Backup
    with open(filepath + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Backup created: demo.py.backup")

    # Fix TensorFlow import
    if 'import tensorflow as tf' in content and 'tensorflow.compat.v1' not in content:
        content = content.replace(
            'import tensorflow as tf',
            'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()'
        )
        print("✓ Fixed TensorFlow import")

    # Write fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Fixed {filepath} successfully!")

except Exception as e:
    print(f"Error: {e}")