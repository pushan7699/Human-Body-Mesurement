"""
Retrain weight model using ONLY real samples (no synthetic data)
This should fix the weight prediction issue
"""

import json
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print("="*60)
print("RETRAINING WEIGHT MODEL - REAL DATA ONLY")
print("="*60)

# Load training data
with open('real_training_data.json', 'r') as f:
    data = json.load(f)

# Filter ONLY real samples (exclude synthetic)
# Include both 'real_measurement' and 'real_4d_measurement'
real_samples = [s for s in data if 'real' in s['timestamp'] and 'synthetic' not in s['timestamp']]

print(f"\n📊 Total samples in file: {len(data)}")
print(f"✅ Real samples: {len(real_samples)}")
print(f"❌ Synthetic samples (excluded): {len(data) - len(real_samples)}")

if len(real_samples) < 3:
    print("\n⚠️  WARNING: Only {} real samples found!".format(len(real_samples)))
    print("Need at least 3 samples for training. Please collect more real data.")
    exit(1)

print("\n" + "="*60)
print("REAL TRAINING DATA:")
print("="*60)
for s in real_samples:
    print(f"  {s['name']:15s} | H: {s['actual_height']:5.1f} cm | W: {s['actual_weight']:5.1f} kg | Age: {s['age']:2d} | {s['gender']}")

# Prepare training data
X_height = []
X_weight = []
y_height = []
y_weight = []

for sample in real_samples:
    features = sample['features']
    
    # Height features
    height_feat = [
        features.get('raw_height', 0),
        features.get('shoulder_width', 0),
        features.get('chest_width', 0),
        features.get('waist_width', 0),
        features.get('hip_width', 0)
    ]
    X_height.append(height_feat)
    y_height.append(sample['actual_height'])
    
    # Weight features (more complex)
    height_cm = features.get('raw_height', 170)
    shoulder_w = features.get('shoulder_width', 0.4)
    shoulder_d = features.get('shoulder_depth', 0.3)
    chest_w = features.get('chest_width', 0.35)
    chest_d = features.get('chest_depth', 0.25)
    waist_w = features.get('waist_width', 0.3)
    waist_d = features.get('waist_depth', 0.2)
    hip_w = features.get('hip_width', 0.35)
    hip_d = features.get('hip_depth', 0.25)
    
    # Calculate volumes
    shoulder_area = np.pi * (shoulder_w / 2) * (shoulder_d / 2)
    chest_area = np.pi * (chest_w / 2) * (chest_d / 2)
    waist_area = np.pi * (waist_w / 2) * (waist_d / 2)
    hip_area = np.pi * (hip_w / 2) * (hip_d / 2)
    
    torso_height = height_cm * 0.5
    torso_volume = (torso_height / 3) * (shoulder_area + chest_area + np.sqrt(shoulder_area * chest_area))
    torso_volume += (torso_height / 3) * (chest_area + waist_area + np.sqrt(chest_area * waist_area))
    
    lower_height = height_cm * 0.5
    lower_volume = (lower_height / 3) * (hip_area + waist_area + np.sqrt(hip_area * waist_area))
    
    total_volume = torso_volume + lower_volume
    bmi_approx = (chest_w + waist_w) / (height_cm / 100)
    waist_to_hip = waist_w / hip_w if hip_w > 0 else 1.0
    
    weight_feat = [
        features.get('raw_height', 0),
        features.get('shoulder_width', 0),
        features.get('chest_width', 0),
        features.get('waist_width', 0),
        features.get('hip_width', 0),
        features.get('shoulder_depth', 0),
        features.get('chest_depth', 0),
        features.get('waist_depth', 0),
        features.get('hip_depth', 0),
        total_volume,
        torso_volume,
        chest_area,
        waist_area,
        bmi_approx,
        waist_to_hip,
        1 if sample['gender'] == 'M' else 0,
        sample['age']
    ]
    X_weight.append(weight_feat)
    y_weight.append(sample['actual_weight'])

X_height = np.array(X_height)
X_weight = np.array(X_weight)
y_height = np.array(y_height)
y_weight = np.array(y_weight)

print(f"\n📊 Training data shape:")
print(f"   Height: {X_height.shape}")
print(f"   Weight: {X_weight.shape}")

# Train HEIGHT model
print("\n" + "="*60)
print("TRAINING HEIGHT MODEL")
print("="*60)

height_scaler = StandardScaler()
X_height_scaled = height_scaler.fit_transform(X_height)

height_model = Ridge(alpha=1.0)
height_model.fit(X_height_scaled, y_height)

# Evaluate height
height_pred = height_model.predict(X_height_scaled)
height_mae = np.mean(np.abs(height_pred - y_height))
height_rmse = np.sqrt(np.mean((height_pred - y_height)**2))

print(f"✅ Height Model Trained!")
print(f"   MAE: {height_mae:.2f} cm")
print(f"   RMSE: {height_rmse:.2f} cm")

# Train WEIGHT model
print("\n" + "="*60)
print("TRAINING WEIGHT MODEL")
print("="*60)

weight_scaler = StandardScaler()
X_weight_scaled = weight_scaler.fit_transform(X_weight)

weight_model = Ridge(alpha=1.0)
weight_model.fit(X_weight_scaled, y_weight)

# Evaluate weight
weight_pred = weight_model.predict(X_weight_scaled)
weight_mae = np.mean(np.abs(weight_pred - y_weight))
weight_rmse = np.sqrt(np.mean((weight_pred - y_weight)**2))

print(f"✅ Weight Model Trained!")
print(f"   MAE: {weight_mae:.2f} kg")
print(f"   RMSE: {weight_rmse:.2f} kg")

# Show predictions vs actual
print("\n" + "="*60)
print("PREDICTIONS vs ACTUAL")
print("="*60)
print(f"{'Name':<15} | {'Actual H':<8} | {'Pred H':<8} | {'Actual W':<8} | {'Pred W':<8} | {'W Error':<8}")
print("-"*80)
for i, sample in enumerate(real_samples):
    h_actual = y_height[i]
    h_pred = height_pred[i]
    w_actual = y_weight[i]
    w_pred = weight_pred[i]
    w_error = w_pred - w_actual
    print(f"{sample['name']:<15} | {h_actual:6.1f} cm | {h_pred:6.1f} cm | {w_actual:6.1f} kg | {w_pred:6.1f} kg | {w_error:+6.1f} kg")

# Save models
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

with open('height_model.pkl', 'wb') as f:
    pickle.dump({
        'model': height_model,
        'scaler': height_scaler
    }, f)
print("✅ Saved: height_model.pkl")

with open('weight_model.pkl', 'wb') as f:
    pickle.dump({
        'model': weight_model,
        'scaler': weight_scaler
    }, f)
print("✅ Saved: weight_model.pkl")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\n🔄 Please restart the web server to use the new models:")
print("   1. Stop the current server (Ctrl+C)")
print("   2. Run: venv_py37\\Scripts\\python.exe web_app.py")
print("\n📊 Expected accuracy:")
print(f"   Height: ±{height_mae:.1f} cm")
print(f"   Weight: ±{weight_mae:.1f} kg")
print("\n⚠️  Note: With only {len(real_samples)} samples, accuracy may vary for new people.")
print("   Collect more real samples to improve accuracy!")
