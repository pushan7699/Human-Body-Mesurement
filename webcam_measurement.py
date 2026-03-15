import cv2
import numpy as np
import os

from demo import main


# =========================
# HEIGHT ESTIMATION
# =========================
def estimate_height_cm(joints3d):
    y = joints3d[:, 1]
    return (y.max() - y.min()) * 100.0


# =========================
# WEIGHT ESTIMATION
# =========================
def estimate_weight_from_height(height_cm):
    height_m = height_cm / 100.0
    bmi = 22.0
    return bmi * (height_m ** 2)


# =========================
# CAPTURE FROM WEBCAM
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera started")
print("Press F = capture FRONT")
print("Press S = capture SIDE")
print("Press B = capture BACK")
print("Press ENTER = process")
print("Press ESC = exit")

front_img = None
side_img = None
back_img = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    cv2.imshow("Webcam Capture", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('f'):
        front_img = frame.copy()
        print("✅ Front image captured")

    elif key == ord('s'):
        side_img = frame.copy()
        print("✅ Side image captured")

    elif key == ord('b'):
        back_img = frame.copy()
        print("✅ Back image captured")

    elif key == 13:  # ENTER
        break

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# CHECK CAPTURE
# =========================
images = [front_img, side_img, back_img]
images = [img for img in images if img is not None]

if len(images) == 0:
    print("❌ No images captured")
    exit()

print(f"Processing {len(images)} view(s)...")

# =========================
# RUN MODEL ON EACH IMAGE
# =========================
heights = []

for img in images:

    joints, verts, cams, joints3d = main(img, None, None)

    if joints3d is None:
        print("⚠️ Pose failed for one view")
        continue

    height_cm = estimate_height_cm(joints3d[0])
    heights.append(height_cm)

# =========================
# FINAL RESULTS
# =========================
if len(heights) == 0:
    print("❌ Pose estimation failed")
    exit()

avg_height = np.mean(heights)
weight = estimate_weight_from_height(avg_height)

print("\n========== FINAL RESULT ==========")
print(f"Height : {avg_height:.1f} cm")
print(f"Weight : {weight:.1f} kg (estimated)")