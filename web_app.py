"""
Flask Web Application for Body Measurement System
Run: venv_py37\Scripts\python.exe web_app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import pickle
import numpy as np
import cv2
import base64
from demo import main as run_hmr_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load ML models
print("Loading ML models...")
with open('height_model.pkl', 'rb') as f:
    height_data = pickle.load(f)

with open('weight_model.pkl', 'rb') as f:
    weight_data = pickle.load(f)

print("✅ Models loaded!")

def extract_features(joints3d, verts):
    """Extract features from SMPL output with STRICT validation for real people only"""
    if joints3d is None or verts is None:
        return None
    
    # Height
    joints = joints3d[0]
    y_values = joints[:, 1]
    raw_height = float(np.max(y_values) - np.min(y_values)) * 100.0
    
    # Validation 1: Check if height is realistic
    if raw_height < 50 or raw_height > 250:
        print(f"⚠️  Invalid height detected: {raw_height:.1f} cm - likely not a real person")
        return None
    
    # Body metrics
    if len(verts.shape) == 3:
        v = verts[0]
    else:
        v = verts
    
    # Validation 2: CRITICAL - Check depth variance (photos are FLAT!)
    z_values = v[:, 2]
    z_std = np.std(z_values)
    z_range = np.max(z_values) - np.min(z_values)
    z_mean = np.mean(z_values)
    
    # Calculate depth variation coefficient
    depth_coef = z_std / (abs(z_mean) + 0.001)
    
    print(f"📊 Depth analysis:")
    print(f"   - Z std: {z_std:.4f}")
    print(f"   - Z range: {z_range:.4f}")
    print(f"   - Z mean: {z_mean:.4f}")
    print(f"   - Depth coefficient: {depth_coef:.4f}")
    
    # STRICT THRESHOLDS: Real people have significant depth variation
    # Photos/pictures will have very small depth values
    if z_std < 0.08:  # Increased from 0.05
        print(f"❌ REJECTED: Flat object detected (z_std={z_std:.4f} < 0.08)")
        print(f"   This appears to be a PHOTO or PICTURE, not a real person!")
        return None
    
    if z_range < 0.25:  # Increased from 0.15
        print(f"❌ REJECTED: Insufficient depth range (z_range={z_range:.4f} < 0.25)")
        print(f"   This appears to be a PHOTO or PICTURE, not a real person!")
        return None
    
    if depth_coef < 0.15:  # New check
        print(f"❌ REJECTED: Low depth variation coefficient (depth_coef={depth_coef:.4f} < 0.15)")
        print(f"   This appears to be a PHOTO or PICTURE, not a real person!")
        return None
    
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    y_range = y_max - y_min
    
    # Validation 3: Check if body range is realistic
    if y_range < 0.5 or y_range > 2.5:
        print(f"⚠️  Invalid body range: {y_range:.2f} m")
        return None
    
    # Validation 4: Check X-axis spread (width)
    x_values = v[:, 0]
    x_range = np.max(x_values) - np.min(x_values)
    x_std = np.std(x_values)
    
    print(f"📊 Width analysis:")
    print(f"   - X range: {x_range:.4f}")
    print(f"   - X std: {x_std:.4f}")
    
    if x_range < 0.2 or x_range > 2.0:
        print(f"❌ REJECTED: Invalid width range (x_range={x_range:.4f})")
        return None
    
    features = {'raw_height': raw_height}
    
    levels = {
        'shoulder': 0.85,
        'chest': 0.70,
        'waist': 0.50,
        'hip': 0.30
    }
    
    valid_features = 0
    depth_measurements = []
    
    for name, ratio in levels.items():
        y_level = y_min + (y_range * ratio)
        level_verts = v[np.abs(v[:, 1] - y_level) < (y_range * 0.08)]
        
        if len(level_verts) > 10:
            width = level_verts[:, 0].max() - level_verts[:, 0].min()
            depth = level_verts[:, 2].max() - level_verts[:, 2].min()
            
            # Validation 5: Check if measurements are realistic
            if 0.1 < width < 1.5 and 0.1 < depth < 1.0:
                features[f'{name}_width'] = abs(float(width))
                features[f'{name}_depth'] = abs(float(depth))
                depth_measurements.append(depth)
                valid_features += 1
                print(f"   ✓ {name}: width={width:.3f}m, depth={depth:.3f}m")
            else:
                print(f"   ✗ {name}: width={width:.3f}m, depth={depth:.3f}m (out of range)")
    
    # Validation 6: Need at least 3 valid body measurements
    if valid_features < 3:
        print(f"❌ REJECTED: Not enough valid features ({valid_features}/4)")
        return None
    
    # Validation 7: Check average depth of body parts
    if len(depth_measurements) > 0:
        avg_depth = np.mean(depth_measurements)
        print(f"📊 Average body depth: {avg_depth:.3f}m")
        
        if avg_depth < 0.15:  # Real people have depth > 15cm
            print(f"❌ REJECTED: Body too thin (avg_depth={avg_depth:.3f}m < 0.15m)")
            print(f"   This appears to be a PHOTO or PICTURE!")
            return None
    
    print(f"✅ ACCEPTED: Valid real person detected with {valid_features}/4 body measurements")
    return features

def predict_measurements(features, age=25, gender='M'):
    """Predict height and weight"""
    # Height
    height_features = np.array([[
        features.get('raw_height', 0),
        features.get('shoulder_width', 0),
        features.get('chest_width', 0),
        features.get('waist_width', 0),
        features.get('hip_width', 0)
    ]])
    
    height_scaled = height_data['scaler'].transform(height_features)
    predicted_height = height_data['model'].predict(height_scaled)[0]
    
    # Calibration adjustment based on actual measurements
    # Model tends to overestimate by ~5-8 cm, apply correction
    height_calibration_factor = 0.95  # Reduce by 5%
    predicted_height = predicted_height * height_calibration_factor
    
    # Weight - calculate volumes
    height_cm = features.get('raw_height', 170)
    shoulder_w = features.get('shoulder_width', 0.4)
    shoulder_d = features.get('shoulder_depth', 0.3)
    chest_w = features.get('chest_width', 0.35)
    chest_d = features.get('chest_depth', 0.25)
    waist_w = features.get('waist_width', 0.3)
    waist_d = features.get('waist_depth', 0.2)
    hip_w = features.get('hip_width', 0.35)
    hip_d = features.get('hip_depth', 0.25)
    
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
    
    weight_features = np.array([[
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
        1 if gender == 'M' else 0,
        age
    ]])
    
    weight_scaled = weight_data['scaler'].transform(weight_features)
    predicted_weight = weight_data['model'].predict(weight_scaled)[0]
    
    return predicted_height, predicted_weight

@app.route('/')
def index():
    """Home page"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Body Measurement System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .camera-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            background: #000;
            position: relative;
        }
        #video {
            width: 100%;
            max-width: 640px;
            border-radius: 10px;
            display: none;
        }
        #canvas {
            display: none;
        }
        .camera-placeholder {
            padding: 60px 20px;
            color: white;
        }
        .camera-icon {
            font-size: 80px;
            margin-bottom: 20px;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-bottom: 10px;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        #preview {
            max-width: 100%;
            max-height: 400px;
            margin: 20px auto;
            display: none;
            border-radius: 10px;
        }
        #results {
            display: none;
            margin-top: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #f8f9ff 0%, #e8eaff 100%);
            border-radius: 15px;
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 15px;
            margin-bottom: 10px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .result-label {
            font-weight: 600;
            color: #667eea;
        }
        .result-value {
            font-size: 1.2em;
            color: #333;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
            border: 2px solid #c33;
            font-weight: 600;
            white-space: pre-line;
        }
        .error::before {
            content: '⚠️ ';
            font-size: 24px;
        }
        .countdown {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 120px;
            color: #4CAF50;
            font-weight: bold;
            text-shadow: 0 0 20px rgba(0,0,0,0.5);
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📏 Body Measurement System</h1>
        <p class="subtitle">AI-powered height and weight estimation from camera</p>
        
        <div class="camera-area">
            <div class="camera-placeholder" id="placeholder">
                <div class="camera-icon">📷</div>
                <p><strong>Click "Open Camera" to start</strong></p>
                <p style="color: #ccc; margin-top: 10px;">Stand 2-3 meters from camera</p>
            </div>
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas"></canvas>
            <div class="countdown" id="countdown"></div>
        </div>
        
        <button class="btn" onclick="openCamera()" id="openCameraBtn">
            📷 Open Camera
        </button>
        
        <button class="btn btn-secondary" onclick="capturePhoto()" id="captureBtn" style="display: none;">
            📸 Capture Photo (3s countdown)
        </button>
        
        <button class="btn" onclick="stopCamera()" id="stopCameraBtn" style="display: none;">
            ⏹️ Stop Camera
        </button>
        
        <img id="preview" />
        
        <div class="form-group">
            <label>Age</label>
            <input type="number" id="age" value="25" min="1" max="120">
        </div>
        
        <div class="form-group">
            <label>Gender</label>
            <select id="gender">
                <option value="M">Male</option>
                <option value="F">Female</option>
            </select>
        </div>
        
        <button class="btn" onclick="measureBody()" id="measureBtn" style="display: none;">
            🎯 Measure Body
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px;">Processing image...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div id="results">
            <h2 style="text-align: center; margin-bottom: 20px; color: #667eea;">📊 Results</h2>
            <div class="result-item">
                <span class="result-label">Height</span>
                <span class="result-value" id="height">-</span>
            </div>
            <div class="result-item">
                <span class="result-label">Weight</span>
                <span class="result-value" id="weight">-</span>
            </div>
            <div class="result-item">
                <span class="result-label">BMI</span>
                <span class="result-value" id="bmi">-</span>
            </div>
        </div>
    </div>
    
    <script>
        let stream = null;
        let capturedBlob = null;
        
        async function openCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: 'user'
                    } 
                });
                
                const video = document.getElementById('video');
                video.srcObject = stream;
                video.style.display = 'block';
                
                document.getElementById('placeholder').style.display = 'none';
                document.getElementById('openCameraBtn').style.display = 'none';
                document.getElementById('captureBtn').style.display = 'block';
                document.getElementById('stopCameraBtn').style.display = 'block';
                document.getElementById('preview').style.display = 'none';
                document.getElementById('measureBtn').style.display = 'none';
                
            } catch (error) {
                showError('Camera access denied: ' + error.message);
            }
        }
        
        async function capturePhoto() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const countdown = document.getElementById('countdown');
            const captureBtn = document.getElementById('captureBtn');
            
            captureBtn.disabled = true;
            
            // 3 second countdown
            for (let i = 3; i > 0; i--) {
                countdown.textContent = i;
                countdown.style.display = 'block';
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            
            countdown.style.display = 'none';
            
            // Capture
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            // Convert to blob
            canvas.toBlob(function(blob) {
                capturedBlob = blob;
                
                // Show preview
                const preview = document.getElementById('preview');
                preview.src = URL.createObjectURL(blob);
                preview.style.display = 'block';
                
                // Show measure button
                document.getElementById('measureBtn').style.display = 'block';
                
                captureBtn.disabled = false;
            }, 'image/jpeg', 0.95);
        }
        
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            document.getElementById('video').style.display = 'none';
            document.getElementById('placeholder').style.display = 'block';
            document.getElementById('openCameraBtn').style.display = 'block';
            document.getElementById('captureBtn').style.display = 'none';
            document.getElementById('stopCameraBtn').style.display = 'none';
        }
        
        async function measureBody() {
            if (!capturedBlob) {
                showError('Please capture a photo first!');
                return;
            }
            
            const age = document.getElementById('age').value;
            const gender = document.getElementById('gender').value;
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('measureBtn').disabled = true;
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('image', capturedBlob, 'capture.jpg');
            formData.append('age', age);
            formData.append('gender', gender);
            
            try {
                const response = await fetch('/measure', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Show results
                    document.getElementById('height').textContent = 
                        data.height + ' cm (' + data.height_inches + ' inches)';
                    document.getElementById('weight').textContent = 
                        data.weight + ' kg (' + data.weight_lbs + ' lbs)';
                    document.getElementById('bmi').textContent = 
                        data.bmi + ' (' + data.bmi_category + ')';
                    
                    document.getElementById('results').style.display = 'block';
                } else {
                    showError(data.error || 'Measurement failed');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('measureBtn').disabled = false;
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.innerHTML = '<strong>ERROR</strong><br>' + message;
            errorDiv.style.display = 'block';
            
            // Scroll to error
            errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    </script>
</body>
</html>
    '''

@app.route('/measure', methods=['POST'])
def measure():
    """Process image and return measurements"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided. Please capture a photo first.'}), 400
        
        image_file = request.files['image']
        age = int(request.form.get('age', 25))
        gender = request.form.get('gender', 'M')
        
        # Read image
        img_bytes = image_file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format. Please try capturing again.'}), 400
        
        # Check image size
        height, width = img.shape[:2]
        if width < 200 or height < 200:
            return jsonify({'error': 'Image too small. Please ensure good camera quality.'}), 400
        
        # Run HMR model
        print("Running HMR model...")
        joints, verts, cams, joints3d = run_hmr_model(img, None, None)
        
        # Check if person was detected
        if joints3d is None or verts is None:
            print("❌ HMR returned None - no person detected")
            return jsonify({
                'error': 'No person detected in image! Please ensure:\n• A person is clearly visible\n• Stand 2-3 meters from camera\n• Good lighting\n• Full body is in frame'
            }), 400
        
        # Additional check: verify joints3d shape
        if joints3d.shape[0] == 0 or joints3d.shape[1] < 19:
            print(f"❌ Invalid joints shape: {joints3d.shape}")
            return jsonify({
                'error': 'Invalid person detection! The image may contain:\n• Photos/pictures instead of real person\n• Multiple people (stand alone)\n• Partial body only\n\nPlease capture again with one person clearly visible.'
            }), 400
        
        # Extract features
        print("Extracting features...")
        features = extract_features(joints3d, verts)
        
        if features is None or len(features) < 5:
            return jsonify({
                'error': '❌ NO REAL PERSON DETECTED!\n\n🚫 This appears to be a PHOTO/PICTURE, not a real person!\n\nCommon issues:\n• Pointing camera at a photo on wall\n• Showing picture on phone/screen\n• Person too far away (>5 meters)\n• Poor lighting or image quality\n• Only partial body visible\n\n✅ To fix:\n• Stand in front of camera YOURSELF\n• Keep distance: 2-3 meters\n• Ensure FULL BODY is visible\n• Use good lighting\n• Stand still during capture\n\n💡 The system detects depth - photos are flat and will be rejected!'
            }), 400
        
        # Predict
        print("Predicting measurements...")
        height, weight = predict_measurements(features, age, gender)
        
        if height is None or weight is None or height < 100 or height > 250 or weight < 30 or weight > 200:
            return jsonify({
                'error': 'Unrealistic measurements detected! Please:\n• Stand 2-3 meters from camera\n• Ensure proper lighting\n• Stand straight and still\n• Try capturing again'
            }), 400
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        
        results = {
            'height': round(height, 1),
            'weight': round(weight, 1),
            'bmi': round(bmi, 1),
            'bmi_category': bmi_category,
            'height_inches': round(height / 2.54, 1),
            'weight_lbs': round(weight * 2.205, 1)
        }
        
        print(f"✅ Results: {results}")
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 BODY MEASUREMENT WEB APPLICATION")
    print("="*60)
    print("\n✅ Starting web server...")
    print("📱 Open browser: http://localhost:5000")
    print("📸 Upload images to get measurements")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
