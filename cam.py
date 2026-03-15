#!/usr/bin/env python3
"""
Camera Display Application using OpenCV and OpenDR
This script captures video from your camera and displays it in real-time.
It includes OpenDR integration for image processing when available.
"""

import cv2
import numpy as np
import sys
import time

# Try to import OpenDR components
try:
    from opendr.engine.data import Image as OpenDRImage
    from opendr.engine.target import BoundingBox
    OPENDR_AVAILABLE = True
    print("OpenDR is available!")
except ImportError as e:
    print(f"OpenDR not available: {e}")
    print("Continuing with OpenCV only...")
    OPENDR_AVAILABLE = False

class CameraDisplay:
    def __init__(self, camera_id=0, window_name="Camera Feed"):
        """
        Initialize camera display
        
        Args:
            camera_id (int): Camera device ID (usually 0 for default camera)
            window_name (str): Name of the display window
        """
        self.camera_id = camera_id
        self.window_name = window_name
        self.cap = None
        self.is_running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def initialize_camera(self):
        """Initialize the camera capture"""
        print(f"Initializing camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully!")
        return True
    
    def convert_to_opendr_image(self, cv_image):
        """Convert OpenCV image to OpenDR Image format"""
        if OPENDR_AVAILABLE:
            try:
                # Convert BGR to RGB (OpenDR typically uses RGB)
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                opendr_img = OpenDRImage(rgb_image)
                return opendr_img
            except Exception as e:
                print(f"Error converting to OpenDR Image: {e}")
                return None
        return None
    
    def process_frame(self, frame):
        """
        Process each frame - add your OpenDR processing here
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            processed_frame: Processed frame for display
        """
        processed_frame = frame.copy()
        
        # Convert to OpenDR format if available
        opendr_image = self.convert_to_opendr_image(frame)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, f"Time: {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add FPS counter
        cv2.putText(processed_frame, f"FPS: {self.current_fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add OpenDR status
        status = "OpenDR: Available" if OPENDR_AVAILABLE else "OpenDR: Not Available"
        cv2.putText(processed_frame, status, 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add frame dimensions
        height, width = frame.shape[:2]
        cv2.putText(processed_frame, f"Size: {width}x{height}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Example: Add a simple edge detection overlay (optional)
        # Uncomment the lines below to see edge detection
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 100, 200)
        # processed_frame[:, :, 0] = np.where(edges > 0, 255, processed_frame[:, :, 0])
        
        return processed_frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
        
        if elapsed_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """Main camera display loop"""
        if not self.initialize_camera():
            return
        
        print("Starting camera feed...")
        print("Press 'q' to quit, 's' to save screenshot, 'f' to toggle fullscreen")
        
        self.is_running = True
        fullscreen = False
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_RESIZABLE)
        
        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow(self.window_name, processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Screenshot saved as {filename}")
                elif key == ord('f'):
                    # Toggle fullscreen
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
                elif key == ord('h'):
                    # Show help
                    print("\nControls:")
                    print("  q - Quit")
                    print("  s - Save screenshot")
                    print("  f - Toggle fullscreen")
                    print("  h - Show this help")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete!")

def main():
    """Main function"""
    print("Camera Display Application")
    print("=" * 30)
    
    # You can change the camera_id if you have multiple cameras
    # 0 is usually the default camera, 1 for external USB camera, etc.
    camera_id = 0
    
    # Create and run camera display
    camera_display = CameraDisplay(camera_id=camera_id)
    camera_display.run()

if __name__ == "__main__":
    main()