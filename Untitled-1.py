"""
Body Measurement System - Console Version with Enhanced Validation
Uses improved validation and measurement extraction
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import joblib
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

@dataclass
class BodyMeasurements:
    height: float
    shoulder_width: float
    chest_width: float
    waist_width: float
    hip_width: float
    torso_length: float
    arm_span: float
    leg_length: float
    head_size: float

    def get_summary(self):
        return f"""
📊 Body Measurements:
  Height: {self.height:.1f} cm
  Shoulder Width: {self.shoulder_width:.1f} cm
  Chest Width: {self.chest_width:.1f} cm
  Waist Width: {self.waist_width:.1f} cm
  Hip Width: {self.hip_width:.1f} cm
  Torso Length: {self.torso_length:.1f} cm
  Arm Span: {self.arm_span:.1f} cm
  Leg Length: {self.leg_length:.1f} cm
"""

class SimpleBodyDetector:
    """
    Simple body detection using OpenCV without MediaPipe
    Uses contour detection and basic computer vision
    """

    def __init__(self):  # Fixed: was _init_
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.person_detected = False

    def detect_person_contour(self, frame):
        """Detect person using background subtraction and contours"""

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, fg_mask

        # Find the largest contour (assumed to be person)
        largest_contour = max(contours, key=cv2.contourArea)

        # Filter out small contours
        if cv2.contourArea(largest_contour) < 5000:
            return None, fg_mask

        self.person_detected = True
        return largest_contour, fg_mask

    def extract_key_points(self, contour, frame_shape):
        """Extract key body points from contour"""
        if contour is None:
            return None

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Get contour points
        contour_points = contour.reshape(-1, 2)

        # Find key points using contour analysis
        key_points = {}

        # Head (topmost point)
        head_point = tuple(contour_points[contour_points[:, 1].argmin()])
        key_points['head'] = head_point

        # Feet (bottommost points)
        bottom_points = contour_points[contour_points[:, 1] > y + h * 0.8]
        if len(bottom_points) > 0:
            # Find left and right foot
            left_foot = tuple(bottom_points[bottom_points[:, 0].argmin()])
            right_foot = tuple(bottom_points[bottom_points[:, 0].argmax()])
            key_points['left_foot'] = left_foot
            key_points['right_foot'] = right_foot
        else:
            # If we can't find feet, use the bottom of the contour
            bottom_point = tuple(contour_points[contour_points[:, 1].argmax()])
            key_points['left_foot'] = (bottom_point[0] - 10, bottom_point[1])
            key_points['right_foot'] = (bottom_point[0] + 10, bottom_point[1])

        # Shoulders (widest point in upper third)
        upper_third = contour_points[contour_points[:, 1] < y + h * 0.4]
        if len(upper_third) > 0:
            # Find widest point in upper region
            shoulder_y = int(np.mean(upper_third[:, 1]))
            shoulder_points = contour_points[
                abs(contour_points[:, 1] - shoulder_y) < h * 0.1
            ]
            if len(shoulder_points) > 0:
                left_shoulder = tuple(shoulder_points[shoulder_points[:, 0].argmin()])
                right_shoulder = tuple(shoulder_points[shoulder_points[:, 0].argmax()])
                key_points['left_shoulder'] = left_shoulder
                key_points['right_shoulder'] = right_shoulder
            else:
                # Estimate shoulder positions if not detected
                key_points['left_shoulder'] = (x, y + h * 0.15)
                key_points['right_shoulder'] = (x + w, y + h * 0.15)
        else:
            # Estimate shoulder positions if upper third is empty
            key_points['left_shoulder'] = (x, y + h * 0.15)
            key_points['right_shoulder'] = (x + w, y + h * 0.15)

        # Hips (widest point in lower third)
        lower_region = contour_points[
            (contour_points[:, 1] > y + h * 0.5) &
            (contour_points[:, 1] < y + h * 0.8)
        ]
        if len(lower_region) > 0:
            hip_y = int(np.mean(lower_region[:, 1]))
            hip_points = contour_points[
                abs(contour_points[:, 1] - hip_y) < h * 0.1
            ]
            if len(hip_points) > 0:
                left_hip = tuple(hip_points[hip_points[:, 0].argmin()])
                right_hip = tuple(hip_points[hip_points[:, 0].argmax()])
                key_points['left_hip'] = left_hip
                key_points['right_hip'] = right_hip
            else:
                # Estimate hip positions if not detected
                key_points['left_hip'] = (x + w * 0.2, y + h * 0.6)
                key_points['right_hip'] = (x + w * 0.8, y + h * 0.6)
        else:
            # Estimate hip positions if lower region is empty
            key_points['left_hip'] = (x + w * 0.2, y + h * 0.6)
            key_points['right_hip'] = (x + w * 0.8, y + h * 0.6)

        # Arms (outermost points in middle region)
        middle_region = contour_points[
            (contour_points[:, 1] > y + h * 0.3) &
            (contour_points[:, 1] < y + h * 0.7)
        ]
        if len(middle_region) > 0:
            left_arm = tuple(middle_region[middle_region[:, 0].argmin()])
            right_arm = tuple(middle_region[middle_region[:, 0].argmax()])
            key_points['left_arm'] = left_arm
            key_points['right_arm'] = right_arm
        else:
            # Estimate arm positions if not detected
            key_points['left_arm'] = (x - 10, y + h * 0.5)
            key_points['right_arm'] = (x + w + 10, y + h * 0.5)

        return key_points

    def draw_keypoints(self, frame, keypoints):
        """Draw detected keypoints on frame"""
        if not keypoints:
            return frame

        colors = {
            'head': (0, 255, 255),  # Yellow
            'left_shoulder': (0, 255, 0),  # Green
            'right_shoulder': (0, 255, 0),
            'left_hip': (255, 0, 0),  # Blue
            'right_hip': (255, 0, 0),
            'left_foot': (0, 0, 255),  # Red
            'right_foot': (0, 0, 255),
            'left_arm': (255, 255, 0),  # Cyan
            'right_arm': (255, 255, 0)
        }

        for point_name, point in keypoints.items():
            color = colors.get(point_name, (255, 255, 255))
            cv2.circle(frame, point, 8, color, -1)
            cv2.putText(frame, point_name.replace('_', ' ').title(),
                       (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)

        # Draw connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_hip', 'right_hip'),
            ('left_foot', 'right_foot')
        ]

        for start, end in connections:
            if start in keypoints and end in keypoints:
                cv2.line(frame, keypoints[start], keypoints[end], (255, 255, 255), 2)

        return frame

class SimpleBodyMeasurements:
    """Extract body measurements from detected keypoints"""

    def __init__(self, reference_height=170):  # Fixed: was _init_
        self.reference_height = reference_height
        self.pixel_to_cm_ratio = None

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  # Fixed: was *2 instead of **2

    def extract_measurements(self, keypoints):
        """Extract body measurements from keypoints"""
        if not keypoints or 'head' not in keypoints:
            return None

        try:
            # Calculate pixel to cm ratio using height
            if 'left_foot' in keypoints and 'right_foot' in keypoints:
                foot_center_y = (keypoints['left_foot'][1] + keypoints['right_foot'][1]) / 2
                body_height_pixels = foot_center_y - keypoints['head'][1]

                if body_height_pixels > 100:  # Reduced minimum reasonable height
                    self.pixel_to_cm_ratio = self.reference_height / body_height_pixels
                else:
                    print(f"⚠ Body height in pixels is too small: {body_height_pixels}")
                    return None
            else:
                print("⚠ Could not detect feet for height calculation")
                return None

            measurements = {}

            # Shoulder width
            if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
                shoulder_width_pixels = self.calculate_distance(
                    keypoints['left_shoulder'], keypoints['right_shoulder']
                )
                measurements['shoulder_width'] = shoulder_width_pixels * self.pixel_to_cm_ratio
                print(f"📏 Shoulder width: {measurements['shoulder_width']:.1f} cm (pixels: {shoulder_width_pixels:.1f}, ratio: {self.pixel_to_cm_ratio:.4f})")
            else:
                print("⚠ Could not detect shoulders, using estimate")
                measurements['shoulder_width'] = self.reference_height * 0.235  # Estimate

            # Hip width
            if 'left_hip' in keypoints and 'right_hip' in keypoints:
                hip_width_pixels = self.calculate_distance(
                    keypoints['left_hip'], keypoints['right_hip']
                )
                measurements['hip_width'] = hip_width_pixels * self.pixel_to_cm_ratio
            else:
                measurements['hip_width'] = measurements['shoulder_width'] * 0.85

            # Torso length
            if ('left_shoulder' in keypoints and 'right_shoulder' in keypoints and
                'left_hip' in keypoints and 'right_hip' in keypoints):

                shoulder_center = (
                    (keypoints['left_shoulder'][0] + keypoints['right_shoulder'][0]) / 2,
                    (keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) / 2
                )
                hip_center = (
                    (keypoints['left_hip'][0] + keypoints['right_hip'][0]) / 2,
                    (keypoints['left_hip'][1] + keypoints['right_hip'][1]) / 2
                )
                torso_length_pixels = self.calculate_distance(shoulder_center, hip_center)
                measurements['torso_length'] = torso_length_pixels * self.pixel_to_cm_ratio
            else:
                measurements['torso_length'] = self.reference_height * 0.32

            # Arm span
            if 'left_arm' in keypoints and 'right_arm' in keypoints:
                arm_span_pixels = self.calculate_distance(
                    keypoints['left_arm'], keypoints['right_arm']
                )
                measurements['arm_span'] = arm_span_pixels * self.pixel_to_cm_ratio
            else:
                measurements['arm_span'] = self.reference_height * 1.02

            # Leg length
            if ('left_hip' in keypoints and 'left_foot' in keypoints and
                'right_hip' in keypoints and 'right_foot' in keypoints):

                left_leg_pixels = self.calculate_distance(keypoints['left_hip'], keypoints['left_foot'])
                right_leg_pixels = self.calculate_distance(keypoints['right_hip'], keypoints['right_foot'])
                avg_leg_pixels = (left_leg_pixels + right_leg_pixels) / 2
                measurements['leg_length'] = avg_leg_pixels * self.pixel_to_cm_ratio
            else:
                measurements['leg_length'] = self.reference_height * 0.47

            # Derived measurements
            measurements['chest_width'] = measurements['shoulder_width'] * 0.85
            measurements['waist_width'] = measurements['hip_width'] * 0.92
            measurements['head_size'] = measurements['shoulder_width'] * 0.32

            # Create BodyMeasurements object
            body_measurements = BodyMeasurements(
                height=self.reference_height,
                shoulder_width=measurements['shoulder_width'],
                chest_width=measurements['chest_width'],
                waist_width=measurements['waist_width'],
                hip_width=measurements['hip_width'],
                torso_length=measurements['torso_length'],
                arm_span=measurements['arm_span'],
                leg_length=measurements['leg_length'],
                head_size=measurements['head_size']
            )

            # Validate measurements with more realistic thresholds
            if self._validate_measurements(body_measurements):
                return body_measurements
            else:
                return None

        except Exception as e:
            print(f"❌ Measurement extraction error: {e}")
            return None

    def _validate_measurements(self, measurements):
        """Validate if measurements are reasonable with more realistic thresholds"""
        # More realistic shoulder width range (15-70 cm)
        if not (15 < measurements.shoulder_width < 70):
            print(f"⚠ Invalid shoulder width: {measurements.shoulder_width:.1f} cm")
            return False

        # More realistic torso length range (30-100 cm)
        if not (30 < measurements.torso_length < 100):
            print(f"⚠ Invalid torso length: {measurements.torso_length:.1f} cm")
            return False

        # Check if any measurement is zero or negative
        for field in ['shoulder_width', 'chest_width', 'waist_width', 'hip_width',
                     'torso_length', 'arm_span', 'leg_length']:
            value = getattr(measurements, field)
            if value <= 0:
                print(f"⚠ Invalid {field}: {value:.1f} cm")
                return False

        return True

class WeightRegressionModel:
    """Weight estimation using a trained regression model"""

    def __init__(self):  # Fixed: was _init_
        self.model = None
        self.scaler = None
        self.model_trained = False
        self.model_file = "weight_regression_model.pkl"
        self.scaler_file = "weight_regression_scaler.pkl"

        # Try to load pre-trained model
        self.load_model()

        # If no model exists, train a new one
        if not self.model_trained:
            print("⚠ No pre-trained model found. Training a new model...")
            self.train_model()
            self.save_model()

    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for the regression model"""
        np.random.seed(42)  # For reproducibility

        # Realistic ranges for body measurements (in cm)
        height = np.random.uniform(150, 200, n_samples)
        shoulder_width = np.random.uniform(35, 55, n_samples) * (height / 170)  # Scale with height
        chest_width = shoulder_width * np.random.uniform(0.8, 0.9, n_samples)
        waist_width = shoulder_width * np.random.uniform(0.7, 0.85, n_samples)
        hip_width = shoulder_width * np.random.uniform(0.85, 0.95, n_samples)
        torso_length = height * np.random.uniform(0.3, 0.35, n_samples)
        arm_span = height * np.random.uniform(0.95, 1.05, n_samples)
        leg_length = height * np.random.uniform(0.45, 0.5, n_samples)

        # Create feature matrix
        X = np.column_stack([height, shoulder_width, chest_width, waist_width,
                            hip_width, torso_length, arm_span, leg_length])

        # Calculate weight based on realistic BMI ranges with some noise
        # Using the formula: weight = BMI * (height/100)^2
        bmi = np.random.uniform(18.5, 30, n_samples)  # From underweight to obese
        weight = bmi * (height / 100) ** 2

        # Add some noise to make it more realistic
        weight += np.random.normal(0, 2, n_samples)

        return X, weight

    def train_model(self):
        """Train the regression model"""
        # Generate synthetic data
        X, y = self.generate_synthetic_data(n_samples=5000)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train a Random Forest regressor
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Model trained successfully")
        print(f"📊 Model Performance:")
        print(f"   - Mean Absolute Error: {mae:.2f} kg")
        print(f"   - R² Score: {r2:.4f}")

        self.model_trained = True

    def save_model(self):
        """Save the trained model and scaler"""
        if self.model_trained:
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            print(f"💾 Model saved to {self.model_file}")

    def load_model(self):
        """Load a pre-trained model and scaler"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            try:
                self.model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                self.model_trained = True
                print(f"💾 Pre-trained model loaded from {self.model_file}")
            except:
                print("❌ Error loading pre-trained model")
                self.model_trained = False

    def estimate_weight(self, measurements):
        """Estimate weight using the trained regression model"""
        if not self.model_trained:
            print("❌ No trained model available")
            return None, None

        # Prepare features for prediction
        features = np.array([[
            measurements.height,
            measurements.shoulder_width,
            measurements.chest_width,
            measurements.waist_width,
            measurements.hip_width,
            measurements.torso_length,
            measurements.arm_span,
            measurements.leg_length
        ]])

        # Scale the features
        features_scaled = self.scaler.transform(features)

        # Make prediction
        estimated_weight = self.model.predict(features_scaled)[0]

        # Calculate confidence based on feature importance and input validity
        confidence = self.calculate_confidence(measurements)

        return estimated_weight, confidence

    def calculate_confidence(self, measurements):
        """Calculate confidence interval based on feature validity"""
        # Check if all measurements are within reasonable ranges
        valid_ranges = [
            (140 <= measurements.height <= 220),
            (30 <= measurements.shoulder_width <= 60),
            (25 <= measurements.chest_width <= 55),
            (20 <= measurements.waist_width <= 50),
            (25 <= measurements.hip_width <= 60),
            (40 <= measurements.torso_length <= 70),
            (130 <= measurements.arm_span <= 220),
            (60 <= measurements.leg_length <= 110)
        ]

        # Calculate percentage of valid measurements
        validity_score = sum(valid_ranges) / len(valid_ranges)

        # Base confidence interval (± kg)
        base_interval = 3.5

        # Adjust based on validity
        confidence_interval = base_interval * (2 - validity_score)

        return confidence_interval

    def get_body_analysis(self, measurements, weight):
        """Get additional body analysis"""
        bmi = weight / ((measurements.height / 100) ** 2)

        # BMI category
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal weight"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        # Body proportions
        proportions = {
            'shoulder_height_ratio': (measurements.shoulder_width / measurements.height) * 100,
            'waist_height_ratio': (measurements.waist_width / measurements.height) * 100,
            'leg_height_ratio': (measurements.leg_length / measurements.height) * 100,
            'armspan_height_ratio': (measurements.arm_span / measurements.height) * 100
        }

        return {
            'bmi': bmi,
            'bmi_category': bmi_category,
            'proportions': proportions
        }

def process_image(image_path, height=170):
    """Process an image and return measurements"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None

    print(f"✅ Loaded image: {image_path}")
    print(f"📏 Using reference height: {height} cm")

    # Create detector and extractor
    detector = SimpleBodyDetector()
    measurements_extractor = SimpleBodyMeasurements(height)
    weight_estimator = WeightRegressionModel()

    # Process image
    contour, fg_mask = detector.detect_person_contour(image)
    keypoints = None
    if contour is not None:
        keypoints = detector.extract_key_points(contour, image.shape)

    if keypoints:
        print(f"✅ Detected {len(keypoints)} body keypoints")

        # Extract measurements
        measurements = measurements_extractor.extract_measurements(keypoints)

        if measurements:
            # Estimate weight
            estimated_weight, confidence = weight_estimator.estimate_weight(measurements)
            analysis = weight_estimator.get_body_analysis(measurements, estimated_weight)

            return {
                'measurements': measurements,
                'weight': estimated_weight,
                'confidence': confidence,
                'analysis': analysis,
                'keypoints': keypoints
            }
        else:
            print("❌ Could not extract measurements from keypoints")
            return None
    else:
        print("❌ No person detected in the image")
        return None

def display_results(results):
    """Display the results in a formatted way"""
    if not results:
        print("❌ No results to display")
        return

    measurements = results['measurements']
    weight = results['weight']
    confidence = results['confidence']
    analysis = results['analysis']

    print("\n" + "="*60)
    print("📊 BODY MEASUREMENT RESULTS")
    print("="*60)

    print(measurements.get_summary())

    print(f"\n⚖ WEIGHT ESTIMATION (Using Trained Model)")
    print("="*60)
    print(f"💡 Estimated Weight: {weight:.1f} ± {confidence:.1f} kg")
    print(f"📊 Weight Range: {weight-confidence:.1f} - {weight+confidence:.1f} kg")
    print(f"📈 BMI: {analysis['bmi']:.1f} ({analysis['bmi_category']})")

    print(f"\n📐 BODY PROPORTIONS")
    print("="*60)
    for ratio_name, ratio_value in analysis['proportions'].items():
        readable_name = ratio_name.replace('_', ' ').title()
        print(f"{readable_name}: {ratio_value:.1f}%")

    print(f"\n⚠ IMPORTANT NOTES")
    print("="*60)
    print("• This method provides estimates for educational purposes only")
    print("• Actual weight varies based on body composition")
    print("• For accurate measurements, consult a healthcare professional")

def main():
    """Main function"""
    print("🤖 Body Measurement System - Console Version")
    print("="*60)
    print("📋 This version uses a trained regression model for weight estimation")
    print("🎯 Best results with solid background and good lighting")

    # Get image path
    image_path = input("Enter the path to your image: ").strip()

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Get height
    height_input = input("Enter your height in cm (140-220, default=170): ").strip()
    if height_input:
        try:
            height = float(height_input)
            if not (140 <= height <= 220):
                print("⚠ Height must be between 140-220 cm. Using default 170 cm.")
                height = 170
        except ValueError:
            print("⚠ Invalid height. Using default 170 cm.")
            height = 170
    else:
        height = 170

    print(f"\n🔄 Processing image...")

    # Process the image
    results = process_image(image_path, height)

    if results:
        display_results(results)

        # Option to save results
        save = input("\n💾 Would you like to save these results to a file? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Enter filename (default: body_measurements.txt): ").strip()
            if not filename:
                filename = "body_measurements.txt"

            try:
                with open(filename, 'w') as f:
                    f.write("Body Measurement Results\n")
                    f.write("="*50 + "\n\n")
                    f.write(results['measurements'].get_summary())
                    f.write(f"\nWeight Estimation: {results['weight']:.1f} ± {results['confidence']:.1f} kg\n")
                    f.write(f"BMI: {results['analysis']['bmi']:.1f} ({results['analysis']['bmi_category']})\n")

                print(f"✅ Results saved to {filename}")
            except Exception as e:
                print(f"❌ Could not save results: {e}")
    else:
        print("❌ Could not process the image. Please try with a different image.")

        print("\n💡 Tips for better results:")
        print("• Use a plain, solid-color background")
        print("• Ensure good, even lighting")
        print("• Stand in a T-pose (arms outstretched)")
        print("• Make sure your entire body is visible from head to toe")
        print("• Wear clothing that contrasts with the background")

if __name__ == "__main__":  # Fixed: was _name_ == "_main_"
    main()