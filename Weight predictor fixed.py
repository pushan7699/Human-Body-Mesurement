import numpy as np


def predict_weight(chest, waist, hip, height):
    """
    Formula-based weight prediction (no ML model)
    Works even with wrong measurements
    """

    print(f"\n📏 Raw inputs: chest={chest:.1f}, waist={waist:.1f}, hip={hip:.1f}, height={height:.1f}")

    # Clamp to reasonable ranges
    chest = np.clip(chest, 60, 140)
    waist = np.clip(waist, 50, 130)
    hip = np.clip(hip, 70, 150)
    height = np.clip(height, 140, 210)

    print(f"📏 After clipping: chest={chest:.1f}, waist={waist:.1f}, hip={hip:.1f}, height={height:.1f}")

    height_m = height / 100

    # Body volume estimation from circumferences
    avg_circ = (chest + waist + hip) / 3
    avg_radius = avg_circ / (2 * np.pi)
    volume_cm3 = np.pi * (avg_radius ** 2) * height

    # Human body density: ~1.05 g/cm³
    weight = (volume_cm3 * 1.05) / 1000

    # Safety bounds based on BMI (18.5 to 35)
    min_weight = 18.5 * (height_m ** 2)
    max_weight = 35.0 * (height_m ** 2)
    weight = np.clip(weight, min_weight, max_weight)

    bmi = weight / (height_m ** 2)

    print(f"💪 Predicted Weight: {weight:.1f} kg (BMI: {bmi:.1f})")

    return round(float(weight), 2)