"""
View the training dataset used for ML model
"""

import json

print("="*80)
print("TRAINING DATASET - BODY MEASUREMENT SYSTEM")
print("="*80)

# Load data
with open('real_training_data.json', 'r') as f:
    data = json.load(f)

# Separate real and synthetic
real_samples = [s for s in data if 'real' in s['timestamp'] and 'synthetic' not in s['timestamp']]
synthetic_samples = [s for s in data if 'synthetic' in s['timestamp']]

print(f"\n📊 DATASET SUMMARY:")
print(f"   Total samples: {len(data)}")
print(f"   ✅ Real samples: {len(real_samples)}")
print(f"   ❌ Synthetic samples: {len(synthetic_samples)} (NOT used in current model)")

print("\n" + "="*80)
print("REAL TRAINING DATA (Used in ML Model)")
print("="*80)
print(f"\n{'#':<3} {'Name':<15} {'Height (cm)':<12} {'Weight (kg)':<12} {'Age':<5} {'Gender':<7} {'BMI':<6}")
print("-"*80)

for i, sample in enumerate(real_samples, 1):
    height_m = sample['actual_height'] / 100
    bmi = sample['actual_weight'] / (height_m ** 2)
    print(f"{i:<3} {sample['name']:<15} {sample['actual_height']:>6.1f} cm    {sample['actual_weight']:>6.1f} kg    "
          f"{sample['age']:<5} {sample['gender']:<7} {bmi:>5.1f}")

# Statistics
heights = [s['actual_height'] for s in real_samples]
weights = [s['actual_weight'] for s in real_samples]

print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"Height Range: {min(heights):.1f} - {max(heights):.1f} cm (avg: {sum(heights)/len(heights):.1f} cm)")
print(f"Weight Range: {min(weights):.1f} - {max(weights):.1f} kg (avg: {sum(weights)/len(weights):.1f} kg)")
print(f"Age Range: {min(s['age'] for s in real_samples)} - {max(s['age'] for s in real_samples)} years")
print(f"Gender: All Male")

print("\n" + "="*80)
print("NOTES")
print("="*80)
print("• The ML model is trained ONLY on these 8 real samples")
print("• Synthetic samples (500) are NOT used in the current model")
print("• All samples are males aged 16-26 years")
print("• Weight predictions are most accurate for people in 58-90 kg range")
print("• Height predictions are most accurate for people in 165-184 cm range")
print("• To improve accuracy: collect more diverse samples (females, different ages, body types)")
print("\n" + "="*80)

# Show sample features
print("\nSAMPLE FEATURE EXTRACTION (First person):")
print("="*80)
sample = real_samples[0]
print(f"Name: {sample['name']}")
print(f"Actual: {sample['actual_height']:.1f} cm, {sample['actual_weight']:.1f} kg")
print(f"\nExtracted Features:")
for key, value in sample['features'].items():
    if isinstance(value, (int, float)):
        print(f"  {key:20s}: {value:>8.4f}")

print("\n" + "="*80)
print("To add more samples, run: venv_py37\\Scripts\\python.exe add_real_sample_4d.py")
print("="*80)
