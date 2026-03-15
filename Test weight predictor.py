"""
test_weight_predictor.py
Run this to check if your weight_predictor.py is fixed
"""

print("Testing weight_predictor.py...")
print("=" * 60)

try:
    from weight_predictor import predict_weight

    # Test with realistic measurements
    chest = 95
    waist = 85
    hip = 98
    height = 170

    print(f"\nTest inputs:")
    print(f"  Chest:  {chest} cm")
    print(f"  Waist:  {waist} cm")
    print(f"  Hip:    {hip} cm")
    print(f"  Height: {height} cm")

    weight = predict_weight(chest, waist, hip, height)

    print(f"\nPredicted weight: {weight} kg")

    # Check if it's using the old method (fixed BMI)
    expected_old = 25.0 * (height / 100) ** 2  # 72.25 kg

    if abs(weight - expected_old) < 1:
        print("\n❌ PROBLEM: Still using OLD weight_predictor.py!")
        print("   Weight is based only on height (BMI=25.0)")
        print("   You need to replace weight_predictor.py with the fixed version")
    else:
        print("\n✅ SUCCESS: Using NEW weight_predictor.py!")
        print("   Weight is calculated from actual measurements")

except Exception as e:
    print(f"\n❌ ERROR: {e}")

print("=" * 60)