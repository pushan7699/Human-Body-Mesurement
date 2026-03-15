"""
diagnose_model.py
Check what's wrong with your ML model
"""

import numpy as np
import joblib

print("=" * 60)
print("ML MODEL DIAGNOSTIC")
print("=" * 60)

# Load model
try:
    model = joblib.load("weight_regression_model.pkl")
    scaler = joblib.load("weight_regression_scaler.pkl")
    print("✅ Model loaded successfully\n")

    # Check what the model expects
    print("Model Information:")
    print(f"  Type: {type(model).__name__}")

    if hasattr(model, 'n_features_in_'):
        print(f"  Expected features: {model.n_features_in_}")

    if hasattr(scaler, 'n_features_in_'):
        print(f"  Scaler expects: {scaler.n_features_in_} features")

    if hasattr(scaler, 'mean_'):
        print(f"\n  Training data means:")
        feature_names = ['chest', 'waist', 'hip', 'height', 'shoulder', 'torso', 'leg', 'arm']
        for i, (name, mean) in enumerate(zip(feature_names, scaler.mean_)):
            print(f"    {name:12s}: {mean:6.2f}")

    if hasattr(scaler, 'scale_'):
        print(f"\n  Training data std devs:")
        for i, (name, scale) in enumerate(zip(feature_names, scaler.scale_)):
            print(f"    {name:12s}: {scale:6.2f}")

    # Test prediction with NORMAL values
    print("\n" + "=" * 60)
    print("TEST 1: Normal adult measurements")
    print("=" * 60)

    test_features = np.array([[
        95,  # chest
        85,  # waist
        98,  # hip
        175,  # height
        45,  # shoulder
        61.25,  # torso (175 * 0.35)
        84,  # leg (175 * 0.48)
        64.75  # arm (175 * 0.37)
    ]])

    print("Input features:")
    for name, val in zip(feature_names, test_features[0]):
        print(f"  {name:12s}: {val:6.2f}")

    scaled = scaler.transform(test_features)
    weight = model.predict(scaled)[0]

    print(f"\nPredicted weight: {weight:.2f} kg")

    # Check if prediction is reasonable
    height_m = 1.75
    expected_range = (18.5 * height_m ** 2, 35 * height_m ** 2)

    if expected_range[0] <= weight <= expected_range[1]:
        print(f"✅ Weight is reasonable (BMI range: {expected_range[0]:.1f} - {expected_range[1]:.1f} kg)")
    else:
        print(f"❌ Weight is UNREASONABLE (should be {expected_range[0]:.1f} - {expected_range[1]:.1f} kg)")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print("The ML model expects measurements in normal human ranges (60-140cm)")
    print("Your current measurements (360cm chest) are WAY outside this range")
    print("\nRECOMMENDATION: Don't use ML model until measurements are fixed")

except FileNotFoundError:
    print("❌ Model files not found!")
    print("   Missing: weight_regression_model.pkl or weight_regression_scaler.pkl")
except Exception as e:
    print(f"❌ Error: {e}")

print("=" * 60)