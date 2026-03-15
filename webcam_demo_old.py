import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from src.util import image as img_util
from src.RunModel import RunModel


def extract_measurements_direct(frame, height_cm):

    print("\n" + "=" * 60)
    print("DIRECT IMAGE MEASUREMENT (DIAGNOSTIC)")
    print("=" * 60)

    if frame is None:
        print("❌ No frame provided!")
        return None

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Save debug image
    cv2.imwrite('debug_frame.jpg', frame)
    print("✓ Saved debug_frame.jpg")

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite('debug_edges.jpg', edges)
    print("✓ Saved debug_edges.jpg")

    # Dilate edges
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    cv2.imwrite('debug_dilated.jpg', dilated)
    print("✓ Saved debug_dilated.jpg")

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")

    if not contours:
        print("❌ No contours found!")
        return None

    # Get largest contour
    person_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(person_contour)
    print(f"Largest contour area: {contour_area}")

    # Create mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [person_contour], -1, 255, -1)
    cv2.imwrite('debug_mask.jpg', mask)
    print("✓ Saved debug_mask.jpg")

    # Find body bounds
    person_points = np.where(mask > 0)
    if len(person_points[0]) == 0:
        print("❌ Empty mask!")
        return None

    top_y = person_points[0].min()
    bottom_y = person_points[0].max()
    left_x = person_points[1].min()
    right_x = person_points[1].max()

    body_height_px = bottom_y - top_y
    body_width_px = right_x - left_x

    print(f"Body bounds: top={top_y}, bottom={bottom_y}, height={body_height_px}px")
    print(f"Body width: {body_width_px}px")

    if body_height_px == 0:
        print("❌ Zero body height!")
        return None

    px_to_cm = height_cm / body_height_px
    print(f"Pixel to CM ratio: {px_to_cm:.4f}")

    measurements = {}
    measurements['height'] = height_cm

    def measure_width_at_ratio(y_ratio, label=""):
        y_pos = int(top_y + body_height_px * y_ratio)
        if y_pos >= h or y_pos < 0:
            return 0

        row = mask[y_pos, :]
        white_pixels = np.where(row > 0)[0]

        if len(white_pixels) == 0:
            return 0

        width_px = white_pixels[-1] - white_pixels[0]
        width_cm = width_px * px_to_cm

        print(f"  {label:12} @ y={y_ratio:.2f}: {width_px}px = {width_cm:.2f}cm")

        return width_cm

    print("\nMeasuring widths:")

    # Shoulders (15% from top)
    shoulder_width = measure_width_at_ratio(0.15, "Shoulders")
    measurements['shoulder width'] = shoulder_width

    # Chest (28%)
    chest_width = measure_width_at_ratio(0.28, "Chest")
    # Estimate circumference (ellipse approximation)
    measurements['chest'] = chest_width * 2.8  # Rough multiplier

    # Waist (50-55%, find minimum)
    print("\nFinding waist (narrowest point):")
    waist_widths = []
    for r in np.linspace(0.48, 0.58, 11):
        w = measure_width_at_ratio(r, f"Waist {r:.2f}")
        if w > 10:  # Valid measurement
            waist_widths.append(w)

    if waist_widths:
        waist_width = min(waist_widths)
        measurements['waist'] = waist_width * 2.7
        print(f"✓ Waist width selected: {waist_width:.2f}cm → circumference: {measurements['waist']:.2f}cm")
    else:
        measurements['waist'] = measurements['chest'] * 0.85

    # Belly (60%)
    belly_width = measure_width_at_ratio(0.60, "Belly")
    measurements['belly'] = belly_width * 2.75

    # Hips (68-75%, find maximum)
    print("\nFinding hips (widest point):")
    hip_widths = []
    for r in np.linspace(0.68, 0.75, 8):
        w = measure_width_at_ratio(r, f"Hips {r:.2f}")
        if w > 10:
            hip_widths.append(w)

    if hip_widths:
        hip_width = max(hip_widths)
        measurements['hips'] = hip_width * 2.85
        print(f"✓ Hip width selected: {hip_width:.2f}cm → circumference: {measurements['hips']:.2f}cm")
    else:
        measurements['hips'] = measurements['waist'] * 1.1

    # Thigh (78%)
    thigh_width = measure_width_at_ratio(0.78, "Thigh")
    measurements['thigh'] = thigh_width * 1.4  # Single leg

    # Estimates
    measurements['neck'] = shoulder_width * 0.75
    measurements['wrist'] = measurements['waist'] * 0.165
    measurements['ankle'] = height_cm * 0.13
    measurements['arm length'] = height_cm * 0.37

    print("\n" + "=" * 60)
    print("FINAL MEASUREMENTS:")
    print("=" * 60)
    for key, value in sorted(measurements.items()):
        if isinstance(value, (int, float)):
            print(f"{key:18}: {value:6.2f} cm")
    print("=" * 60)

    return measurements


def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = float(224) / max(img.shape[:2])
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)[::-1]
    crop, _ = img_util.scale_and_crop(img, scale, center, 224)
    crop = 2 * ((crop / 255.0) - 0.5)
    return np.expand_dims(crop, 0)


def main():
    print("=" * 60)
    print("DIAGNOSTIC MODE - TESTING IMAGE MEASUREMENTS")
    print("=" * 60)

    sess = tf.Session()
    model = RunModel(sess=sess)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n📷 Press SPACE to capture and analyze")
    print("Press ESC to quit\n")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press SPACE to capture", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            captured_frame = frame.copy()
            print("\n✅ Frame captured! Processing...")
            break

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        print("❌ No frame captured")
        sess.close()
        return

    # Get height input
    try:
        height = float(input("\nEnter your height in cm (e.g., 170): ").strip())
    except:
        height = 170
        print(f"Using default height: {height} cm")

    # Extract measurements DIRECTLY from image
    print("\n" + "=" * 60)
    print("EXTRACTING MEASUREMENTS FROM IMAGE")
    print("=" * 60)

    measurements = extract_measurements_direct(captured_frame, height)

    if measurements:
        print("\n✅ SUCCESS! Check the debug images:")
        print("  - debug_frame.jpg (original)")
        print("  - debug_edges.jpg (edge detection)")
        print("  - debug_dilated.jpg (dilated edges)")
        print("  - debug_mask.jpg (person mask)")

        # Simple weight estimate
        h_m = height / 100.0
        waist_m = measurements.get('waist', 80) / 100.0

        # BMI-based with waist adjustment
        bmi_base = 22.0
        waist_to_height = waist_m / h_m
        if waist_to_height > 0.55:
            bmi_base += 3
        elif waist_to_height > 0.50:
            bmi_base += 1.5

        weight = bmi_base * (h_m ** 2)
        bmi = weight / (h_m ** 2)

        print("\n" + "=" * 60)
        print("WEIGHT ESTIMATE:")
        print(f"Estimated Weight: {weight:.1f} kg")
        print(f"BMI: {bmi:.1f}")
        print("=" * 60)
    else:
        print("\n❌ FAILED to extract measurements")
        print("Check the debug images to see what went wrong")

    sess.close()


if __name__ == "__main__":
    main()