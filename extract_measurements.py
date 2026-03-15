# QUICK FIX FOR YOUR MEASUREMENT ISSUES
# ======================================

## Problem 1: Wrong Pixel-to-CM Scaling

Your
current
code
calculates:
```python
body_px = bottom_y - top_y  # This includes background!
px_to_cm = height_cm / body_px
```

** Issue: ** It
's measuring from top of background to bottom, not just the person.

** Fix in extract_measurements.py
around
line
75: **
```python
# OLD CODE:
# ys_mask, xs_mask = np.where(mask > 0)
# top_y = ys_mask.min()
# bottom_y = ys_mask.max()
# body_px = bottom_y - top_y

# NEW CODE:
ys_mask, xs_mask = np.where(mask > 0)

# Filter to person area only (exclude sparse background pixels)
# Group pixels by row density
row_densities = []
for y in range(h):
    row_pixels = np.sum(mask[y] > 0)
    row_densities.append(row_pixels)

# Find first and last rows with substantial person pixels (>20% of max)
max_density = max(row_densities)
threshold = max_density * 0.2

top_y = None
bottom_y = None

for y in range(h):
    if row_densities[y] > threshold:
        if top_y is None:
            top_y = y
        bottom_y = y

if top_y is None or bottom_y is None:
    print("❌ Could not find person boundaries")
    return None

body_px = bottom_y - top_y

# Additional check: body should be at least 40% of image height
if body_px < 0.4 * h:
    print(f"❌ Body too small: {body_px}px < {0.4 * h}px")
    return None
```

## Problem 2: GrabCut Segmentation Failing

Your
GrabCut is including
too
much
background.

** Fix in extract_measurements.py
around
line
55: **
```python
# OLD CODE:
# rect = (x1, y1, x2 - x1, y2 - y1)

# NEW CODE - Make initial rectangle smaller (only center 70% of joint bounds):
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2
width = int((x2 - x1) * 0.7)
height = int((y2 - y1) * 0.7)

rect = (
    center_x - width // 2,
    center_y - height // 2,
    width,
    height
)

# Also increase iterations for better segmentation:
cv2.grabCut(frame, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)  # 8 instead of 5
```

## Problem 3: Wrong Weight Calculation

Your
code is using
a
fixed
BMI:
```python
bmi = 22.0  # Always 22!
weight = bmi * (height / 100) ** 2
```

** Fix
at
the
end
of
extract_measurements.py: **
```python
# Replace the entire weight calculation section with:

if IMPROVED_PREDICTOR:
    result = predict_weight_improved(height, measurements, gender)
    print(f"💪 Predicted Weight: {result['weight_kg']:.1f} kg")
    print(f"📊 BMI: {result['bmi']:.1f} ({result['bmi_category']})")
    measurements["weight_kg"] = result["weight_kg"]
    measurements["bmi"] = result["bmi"]
else:
    # Use actual measurements for weight estimation
    height_m = height / 100
    chest = measurements.get("chest", 90)
    waist = measurements.get("waist", 80)
    hips = measurements.get("hips", 95)

    # Volume-based weight estimation
    avg_circ = (chest + waist + hips) / 3
    avg_radius = avg_circ / (2 * np.pi)
    volume_cm3 = np.pi * (avg_radius ** 2) * height

    # Human body density ~1.05 g/cm³
    weight = (volume_cm3 * 1.05) / 1000

    # Clamp to reasonable range
    weight = np.clip(weight, 18.5 * height_m ** 2, 35 * height_m ** 2)

    bmi = weight / (height_m ** 2)

    print(f"💪 Estimated Weight: {weight:.1f} kg")
    print(f"📊 BMI: {bmi:.1f}")
    measurements["weight_kg"] = weight
    measurements["bmi"] = bmi
```

## Complete Fixed Section for extract_measurements_from_image()

Here
's the complete fixed function you can replace in your file:

```python


def extract_measurements_from_image(frame, joints3d, height_cm):
    """Fixed version with proper scaling"""
    h, w = frame.shape[:2]

    joints = joints3d[0]
    xs = np.clip((joints[:, 0] * w).astype(int), 0, w - 1)
    ys = np.clip((joints[:, 1] * h).astype(int), 0, h - 1)

    # Bounding box from joints - but make it tighter
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Smaller padding
    pad_x = int(0.10 * (x2 - x1))
    pad_y = int(0.05 * (y2 - y1))

    x1 = max(0, x1 - pad_x)
    x2 = min(w, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y)

    # GrabCut with tighter initial rectangle
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    # Make rect 70% of joint bounds (centered)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    rect_w = int((x2 - x1) * 0.7)
    rect_h = int((y2 - y1) * 0.7)

    rect = (
        center_x - rect_w // 2,
        center_y - rect_h // 2,
        rect_w,
        rect_h
    )

    cv2.grabCut(frame, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

    # Find actual person boundaries (filter sparse pixels)
    row_counts = np.sum(mask > 0, axis=1)
    max_count = row_counts.max()
    threshold = max_count * 0.2

    person_rows = np.where(row_counts > threshold)[0]

    if len(person_rows) < 100:
        print("❌ Person segmentation failed")
        return None

    top_y = person_rows[0]
    bottom_y = person_rows[-1]
    body_px = bottom_y - top_y

    # Check if full body visible
    if body_px < 0.4 * h:
        print(f"❌ Body too small: {body_px}px < {0.4 * h:.0f}px minimum")
        return None

    px_to_cm = height_cm / body_px

    print(f"✓ Body height: {body_px}px = {height_cm}cm")
    print(f"✓ Scaling: {px_to_cm:.4f} cm/px")

    def width_at(r):
        y = int(top_y + body_px * r)
        if y < 0 or y >= h:
            return 0
        row = mask[y]
        xs = np.where(row > 0)[0]
        if len(xs) < 10:
            return 0
        return (xs[-1] - xs[0]) * px_to_cm

    def circumference(width_cm, depth_factor=0.5):
        if width_cm < 5:
            return 0
        a = width_cm / 2
        b = (width_cm * depth_factor) / 2
        h_val = ((a - b) ** 2) / ((a + b) ** 2)
        return np.pi * (a + b) * (1 + (3 * h_val) / (10 + np.sqrt(4 - 3 * h_val)))

    shoulder_w = width_at(0.12)
    chest_w = width_at(0.25)

    # Find narrowest point for waist
    waist_candidates = [width_at(r) for r in np.linspace(0.45, 0.60, 15)]
    waist_candidates = [w for w in waist_candidates if w > 10]
    waist_w = min(waist_candidates) if waist_candidates else chest_w * 0.85

    # Find widest point for hips
    hip_candidates = [width_at(r) for r in np.linspace(0.60, 0.75, 10)]
    hip_w = max(hip_candidates) if hip_candidates else waist_w * 1.1

    measurements = {
        "height": height_cm,
        "shoulder width": shoulder_w,
        "chest": circumference(chest_w, 0.55),
        "waist": circumference(waist_w, 0.50),
        "hips": circumference(hip_w, 0.55),
        "arm length": height_cm * 0.37,
    }

    # Sanity checks
    if measurements["chest"] > 200 or measurements["chest"] < 50:
        print(f"⚠️ Unrealistic chest: {measurements['chest']:.1f}cm")
        return None

    if measurements["waist"] > 180 or measurements["waist"] < 40:
        print(f"⚠️ Unrealistic waist: {measurements['waist']:.1f}cm")
        return None

    return measurements


```

Save
this
to
a
file
called
`measurement_fixes.txt`
for reference!
