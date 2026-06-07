"""
Add real measurement with 4D capture (4 angles)
More accurate training data with width + depth information
"""

import json
import cv2
import numpy as np
from demo import main as run_hmr_model


def extract_features(joints3d, verts):
    """Extract features from SMPL output"""
    
    # Height
    joints = joints3d[0]
    y_values = joints[:, 1]
    raw_height = float(np.max(y_values) - np.min(y_values)) * 100.0
    
    # Body metrics - verts is already (6890, 3), not (1, 6890, 3)
    if len(verts.shape) == 3:
        v = verts[0]
    else:
        v = verts
    
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    y_range = y_max - y_min
    
    features = {'raw_height': raw_height}
    
    levels = {
        'shoulder': 0.85,
        'chest': 0.70,
        'waist': 0.50,
        'hip': 0.30
    }
    
    for name, ratio in levels.items():
        y_level = y_min + (y_range * ratio)
        level_verts = v[np.abs(v[:, 1] - y_level) < (y_range * 0.08)]
        
        if len(level_verts) > 10:
            width = level_verts[:, 0].max() - level_verts[:, 0].min()
            depth = level_verts[:, 2].max() - level_verts[:, 2].min()
            features[f'{name}_width'] = abs(float(width))
            features[f'{name}_depth'] = abs(float(depth))
    
    return features


def capture_4_views(person_name):
    """Capture 4 views for 4D measurement"""
    
    print("\n" + "="*60)
    print(f"4D CAPTURE - {person_name}")
    print("="*60)
    print("\nYou will capture 4 angles:")
    print("  1. FRONT - Face camera")
    print("  2. LEFT SIDE - Turn 90° left")
    print("  3. BACK - Turn around")
    print("  4. RIGHT SIDE - Turn 90° right")
    print("\n💡 Stay 2-3 meters from camera, same distance for all views")
    print("="*60 + "\n")
    
    input("Press ENTER to start...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return None
    
    views = [
        {'name': 'FRONT', 'instruction': 'Face camera', 'img': None, 'features': None},
        {'name': 'LEFT SIDE', 'instruction': 'Turn 90° left', 'img': None, 'features': None},
        {'name': 'BACK', 'instruction': 'Face away', 'img': None, 'features': None},
        {'name': 'RIGHT SIDE', 'instruction': 'Turn 90° right', 'img': None, 'features': None}
    ]
    
    for view in views:
        print(f"\n📸 {view['name']}: {view['instruction']}")
        print("   Press SPACE when ready")
        
        captured = False
        
        while not captured:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                return None
            
            display = frame.copy()
            h, w = display.shape[:2]
            
            cv2.rectangle(display, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(display, f"{person_name} - {view['name']}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display, view['instruction'], 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, "SPACE = Capture  |  ESC = Cancel", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("4D Data Collection", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                # 3 second countdown
                print(f"   ⏱️  Countdown: ", end='', flush=True)
                for i in range(3, 0, -1):
                    print(f"{i}... ", end='', flush=True)
                    
                    # Show countdown on screen
                    countdown_frame = frame.copy()
                    cv2.rectangle(countdown_frame, (0, 0), (w, h), (0, 0, 0), -1)
                    
                    # Draw large countdown number
                    font_scale = 10
                    thickness = 20
                    text = str(i)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    
                    cv2.putText(countdown_frame, text, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                    
                    cv2.imshow("4D Data Collection", countdown_frame)
                    cv2.waitKey(1000)  # Wait 1 second
                
                # Capture after countdown
                ret, final_frame = cap.read()
                if ret:
                    view['img'] = final_frame.copy()
                    captured = True
                    print(f"✅ Captured!")
                    
                    # Flash white
                    white = np.ones_like(final_frame) * 255
                    cv2.imshow("4D Data Collection", white)
                    cv2.waitKey(100)
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    return views


def process_4_views(views):
    """Process all 4 views through HMR"""
    
    print("\n" + "="*60)
    print("PROCESSING 4 VIEWS")
    print("="*60)
    
    for view in views:
        print(f"\n⏳ Processing {view['name']}...")
        
        try:
            joints, verts, cams, joints3d = run_hmr_model(view['img'], None, None)
            
            if joints3d is None or verts is None:
                print(f"   ❌ Failed")
                continue
            
            features = extract_features(joints3d, verts)
            view['features'] = features
            
            print(f"   ✅ Raw height: {features['raw_height']:.1f} cm")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return views


def combine_4d_features(views):
    """Combine features from 4 views into 4D features"""
    
    # Get features from each view
    valid_views = [v for v in views if v['features'] is not None]
    
    if len(valid_views) == 0:
        return None
    
    print(f"\n✅ Successfully processed {len(valid_views)}/4 views")
    
    # Average height from all views
    heights = [v['features']['raw_height'] for v in valid_views]
    avg_height = np.mean(heights)
    
    # Separate front/back (width) from sides (depth)
    front_back = [valid_views[i] for i in [0, 2] if i < len(valid_views)]
    sides = [valid_views[i] for i in [1, 3] if i < len(valid_views)]
    
    combined = {'raw_height': avg_height}
    
    # Width from front/back views
    for level in ['shoulder', 'chest', 'waist', 'hip']:
        widths = []
        for v in front_back:
            if f'{level}_width' in v['features']:
                widths.append(v['features'][f'{level}_width'])
        if widths:
            combined[f'{level}_width'] = np.mean(widths)
    
    # Depth from side views
    for level in ['shoulder', 'chest', 'waist', 'hip']:
        depths = []
        for v in sides:
            # In side view, width becomes depth
            if f'{level}_width' in v['features']:
                depths.append(v['features'][f'{level}_width'])
        if depths:
            combined[f'{level}_depth'] = np.mean(depths)
    
    # If missing depth, estimate from width
    for level in ['shoulder', 'chest', 'waist', 'hip']:
        if f'{level}_depth' not in combined and f'{level}_width' in combined:
            combined[f'{level}_depth'] = combined[f'{level}_width'] * 0.5
    
    print(f"\n📊 4D Features extracted:")
    print(f"   Height: {avg_height:.1f} cm (averaged from {len(valid_views)} views)")
    if 'chest_width' in combined and 'chest_depth' in combined:
        print(f"   ✨ 4D data: Width AND Depth captured!")
    
    return combined


def add_4d_sample():
    """Add a 4D measurement sample"""
    
    print("\n" + "="*60)
    print("ADD 4D REAL MEASUREMENT")
    print("="*60)
    print("\nCapture 4 angles for maximum accuracy!")
    print("="*60 + "\n")
    
    # Get info
    try:
        name = input("Person's name: ").strip()
        actual_height = float(input("Actual height (cm): "))
        actual_weight = float(input("Actual weight (kg): "))
        age = int(input("Age: "))
        gender = input("Gender (M/F): ").strip().upper()
    except ValueError:
        print("❌ Invalid input!")
        return False
    
    print(f"\n✅ Adding: {name}, {actual_height}cm, {actual_weight}kg")
    
    # Capture 4 views
    views = capture_4_views(name)
    
    if views is None:
        return False
    
    # Process
    views = process_4_views(views)
    
    # Combine into 4D features
    features = combine_4d_features(views)
    
    if features is None:
        print("\n❌ Failed to extract features!")
        return False
    
    # Load existing data
    try:
        with open('real_training_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    
    # Add new sample
    new_sample = {
        'name': name,
        'actual_height': actual_height,
        'actual_weight': actual_weight,
        'age': age,
        'gender': gender,
        'features': features,
        'timestamp': 'real_4d_measurement'
    }
    
    data.append(new_sample)
    
    # Save
    with open('real_training_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ 4D sample added!")
    print(f"   Total samples: {len(data)}")
    print(f"   4D samples: {sum(1 for d in data if d['timestamp'] == 'real_4d_measurement')}")
    
    return True


def main():
    """Main program"""
    
    print("\n" + "="*60)
    print("🎯 4D REAL DATA COLLECTION")
    print("="*60)
    print("\nCapture 4 angles for each person:")
    print("  • Front, Left, Back, Right")
    print("  • Captures width AND depth")
    print("  • More accurate training data")
    print("="*60)
    
    print("\n⏳ Loading AI model...")
    
    if add_4d_sample():
        print("\n" + "="*60)
        print("✅ 4D SAMPLE ADDED!")
        print("="*60)
        print("\nNext steps:")
        print("1. Collect more samples (10-15 people recommended)")
        print("   Run: venv_py37\\Scripts\\python.exe add_real_sample_4d.py")
        print("2. Retrain with 4D data:")
        print("   Run: venv_py37\\Scripts\\python.exe train_from_real_data.py")
        print("3. Test accuracy:")
        print("   Run: venv_py37\\Scripts\\python.exe trained_measure.py")
    else:
        print("\n❌ Failed to add sample")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
