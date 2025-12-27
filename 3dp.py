"""
3D Window with perspective correction without markers
Based on Johnny Chung Lee's experiments (2008)

This program creates a "window" effect where the screen acts as a fixed window
through which a 3D scene is viewed. When the body and head move, the perspective 
of the scene changes as if you were looking through a real window, creating an 
illusion of depth and immersion.

The effect works without physical markers, using:
- Body tracking with MediaPipe Pose (main reference)
- Head tracking with MediaPipe Face (fine perspective adjustment)
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- MEDIA PIPE FACE LANDMARKER CONFIGURATION -------------
# Use the modern version of MediaPipe with the face model
# This is more accurate than the old version
model_path = "face_landmarker.task"  # the model you need to download

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

face_landmarker = FaceLandmarker.create_from_options(face_options)

# ---------- CAMERA AND CAMERA MATRIX -------------
# Use camera 1 directly
camera_id = 0
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    print(f"ERROR: Could not open camera {camera_id}!")
    exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened: {w}x{h}")

if w == 0 or h == 0:
    print("ERROR: Camera does not return valid dimensions!")
    cap.release()
    exit(1)

focal_length = w
cam_matrix = np.array([[focal_length, 0, w/2],
                       [0, focal_length, h/2],
                       [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

# Get full screen resolution
# Use common maximum dimensions or adjust according to your screen
screen_width = 1920  # Adjust according to your screen
screen_height = 1080  # Adjust according to your screen

# Create main window that fills the entire screen (but within a window)
# 3D window effect based on Johnny Chung Lee's 2008 experiments
window_name = "3D Window - Body & Head Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# Set window size
cv2.resizeWindow(window_name, screen_width, screen_height)
# Position window at top left
cv2.moveWindow(window_name, 0, 0)
# Ensure window stays as a window (not fullscreen)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)

# No slider needed - depth changes automatically according to head movement

# Use the same figure as box.py
# Base size factor of the background rectangle (70% of screen)
base_size_factor = 0.7

# Screen center
center_x = screen_width / 2
center_y = screen_height / 2

# Function to normalize lines (same as box.py)
def normalize_line(x1, y1, x2, y2):
    """Normalizes a line to avoid duplicates (smallest point first)"""
    if (x1, y1) < (x2, y2):
        return (int(x1), int(y1), int(x2), int(y2))
    else:
        return (int(x2), int(y2), int(x1), int(y1))

# Function to build lines based on box.py, but with background rectangle transformation
def build_lines_array(rect_points_transformed):
    """Builds an array with all unique lines to draw
    rect_points_transformed: 4 points of the background rectangle transformed according to person's movement
    perspective_offset_x, perspective_offset_y: perspective offset for line extensions
    """
    lines_set = set()
    
    # 1. Background rectangle outline
    for i in range(4):
        pt1 = rect_points_transformed[i]
        pt2 = rect_points_transformed[(i + 1) % 4]
        line = normalize_line(pt1[0], pt1[1], pt2[0], pt2[1])
        lines_set.add(line)
    
    # 2. Calculate edge inclinations for vertical lines
    left_top_dx = 0 - rect_points_transformed[0][0]
    left_top_dy = 0 - rect_points_transformed[0][1]
    left_bottom_dx = 0 - rect_points_transformed[3][0]
    left_bottom_dy = screen_height - rect_points_transformed[3][1]
    
    right_top_dx = screen_width - rect_points_transformed[1][0]
    right_top_dy = 0 - rect_points_transformed[1][1]
    right_bottom_dx = screen_width - rect_points_transformed[2][0]
    right_bottom_dy = screen_height - rect_points_transformed[2][1]
    
    # 3. Vertical lines of the background rectangle and their extensions
    for i in range(1, 8):  # 7 vertical lines
        # Calculate relative position within the transformed rectangle
        t = (i - 1) / 7.0 if 7 > 0 else 0
        
        # Interpolate between transformed rectangle points
        x_top = rect_points_transformed[0][0] + t * (rect_points_transformed[1][0] - rect_points_transformed[0][0])
        y_top = rect_points_transformed[0][1] + t * (rect_points_transformed[1][1] - rect_points_transformed[0][1])
        x_bottom = rect_points_transformed[3][0] + t * (rect_points_transformed[2][0] - rect_points_transformed[3][0])
        y_bottom = rect_points_transformed[3][1] + t * (rect_points_transformed[2][1] - rect_points_transformed[3][1])
        
        # Vertical line inside the rectangle
        line = normalize_line(x_top, y_top, x_bottom, y_bottom)
        lines_set.add(line)
        
        # Interpolate inclination between left and right for the top part
        top_dx = left_top_dx + t * (right_top_dx - left_top_dx)
        top_dy = left_top_dy + t * (right_top_dy - left_top_dy)
        if top_dy != 0:
            top_factor = -y_top / top_dy
            top_x_end = x_top + top_dx * top_factor
        else:
            top_x_end = x_top
        
        # Interpolate inclination between left and right for the bottom part
        bottom_dx = left_bottom_dx + t * (right_bottom_dx - left_bottom_dx)
        bottom_dy = left_bottom_dy + t * (right_bottom_dy - left_bottom_dy)
        if bottom_dy != 0:
            bottom_factor = (screen_height - y_bottom) / bottom_dy
            bottom_x_end = x_bottom + bottom_dx * bottom_factor
        else:
            bottom_x_end = x_bottom
        
        # Top extension
        line = normalize_line(x_top, y_top, top_x_end, 0)
        lines_set.add(line)
        
        # Bottom extension
        line = normalize_line(x_bottom, y_bottom, bottom_x_end, screen_height)
        lines_set.add(line)
    
    # 4. Calculate edge inclinations for horizontal lines
    top_left_dx = 0 - rect_points_transformed[0][0]
    top_left_dy = 0 - rect_points_transformed[0][1]
    top_right_dx = screen_width - rect_points_transformed[1][0]
    top_right_dy = 0 - rect_points_transformed[1][1]
    
    bottom_left_dx = 0 - rect_points_transformed[3][0]
    bottom_left_dy = screen_height - rect_points_transformed[3][1]
    bottom_right_dx = screen_width - rect_points_transformed[2][0]
    bottom_right_dy = screen_height - rect_points_transformed[2][1]
    
    # 5. Horizontal lines of the background rectangle and their extensions
    for j in range(1, 8):  # 7 horizontal lines
        # Calculate relative position within the transformed rectangle
        t = (j - 1) / 7.0 if 7 > 0 else 0
        
        # Interpolate between transformed rectangle points
        x_left = rect_points_transformed[0][0] + t * (rect_points_transformed[3][0] - rect_points_transformed[0][0])
        y_left = rect_points_transformed[0][1] + t * (rect_points_transformed[3][1] - rect_points_transformed[0][1])
        x_right = rect_points_transformed[1][0] + t * (rect_points_transformed[2][0] - rect_points_transformed[1][0])
        y_right = rect_points_transformed[1][1] + t * (rect_points_transformed[2][1] - rect_points_transformed[1][1])
        
        # Horizontal line inside the rectangle
        line = normalize_line(x_left, y_left, x_right, y_right)
        lines_set.add(line)
        
        # Interpolate inclination between top and bottom for the left part
        left_dx = top_left_dx + t * (bottom_left_dx - top_left_dx)
        left_dy = top_left_dy + t * (bottom_left_dy - top_left_dy)
        if left_dx != 0:
            left_factor = -x_left / left_dx
            left_y_end = y_left + left_dy * left_factor
        else:
            left_y_end = y_left
        
        # Interpolate inclination between top and bottom for the right part
        right_dx = top_right_dx + t * (bottom_right_dx - top_right_dx)
        right_dy = top_right_dy + t * (bottom_right_dy - top_right_dy)
        if right_dx != 0:
            right_factor = (screen_width - x_right) / right_dx
            right_y_end = y_right + right_dy * right_factor
        else:
            right_y_end = y_right
        
        # Left extension
        line = normalize_line(x_left, y_left, 0, left_y_end)
        lines_set.add(line)
        
        # Right extension
        line = normalize_line(x_right, y_right, screen_width, right_y_end)
        lines_set.add(line)
    
    # 6. 4 lines from rectangle edges (3D rectangle edges)
    line = normalize_line(rect_points_transformed[0][0], rect_points_transformed[0][1], 0, 0)
    lines_set.add(line)
    
    line = normalize_line(rect_points_transformed[1][0], rect_points_transformed[1][1], screen_width, 0)
    lines_set.add(line)
    
    line = normalize_line(rect_points_transformed[2][0], rect_points_transformed[2][1], screen_width, screen_height)
    lines_set.add(line)
    
    line = normalize_line(rect_points_transformed[3][0], rect_points_transformed[3][1], 0, screen_height)
    lines_set.add(line)
    
    # 7. Intermediate rectangles (outlines only)
    for layer in range(1, 8):  # 7 additional rectangles
        t = layer / 8.0
        
        # Interpolate between transformed background rectangle and window edges
        layer_points = []
        for i in range(4):
            x = rect_points_transformed[i][0] + t * ([0, screen_width, screen_width, 0][i] - rect_points_transformed[i][0])
            y = rect_points_transformed[i][1] + t * ([0, 0, screen_height, screen_height][i] - rect_points_transformed[i][1])
            layer_points.append([x, y])
        
        # Outline of this layer's rectangle
        for i in range(4):
            pt1 = layer_points[i]
            pt2 = layer_points[(i + 1) % 4]
            line = normalize_line(pt1[0], pt1[1], pt2[0], pt2[1])
            lines_set.add(line)
    
    return list(lines_set)

# All points and lines have been created in the previous section

# Define box faces with colors
# Colors: BGR format (Blue, Green, Red)
COLOR_BACK = (50, 50, 50)      # Very light gray for back face
COLOR_SIDES = (180, 100, 50)   # Very light blue for sides (BGR: blue dominant)
COLOR_TOP_BOTTOM = (50, 180, 180)  # Light yellow for floor and ceiling (BGR: yellow = green+red)

# Box faces (vertex indices)
box_faces = {
    'back': [4, 5, 6, 7],      # Back face (rear walls)
    'left': [4, 0, 3, 7],      # Left side
    'right': [1, 5, 6, 2],     # Right side
    'top': [3, 2, 6, 7],       # Ceiling
    'bottom': [0, 1, 5, 4]     # Floor
}

# No green cube needed, only the 3D box

# Camera matrix for full screen (centered)
# Use an appropriate focal length for the screen to create realistic perspective
# Focal length determines viewing angle and perspective
# Larger focal length = less distortion, more natural perspective
screen_focal_length = max(screen_width, screen_height) * 1.2
screen_cam_matrix = np.array([[screen_focal_length, 0, screen_width/2],
                              [0, screen_focal_length, screen_height/2],
                              [0, 0, 1]], dtype=np.float32)

# Camera window size (bottom right corner)
camera_window_width = 320
camera_window_height = 240

# Function to draw dashed lines
def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=20):
    """Draws a dashed line between two points"""
    # Verify that points are valid
    if pt1 is None or pt2 is None:
        return
    
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    
    # Verify that points are within the image
    h, w = img.shape[:2]
    if x1 < 0 or x1 >= w or y1 < 0 or y1 >= h:
        if x2 < 0 or x2 >= w or y2 < 0 or y2 >= h:
            return  # Both points outside
    
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if dist < dash_length:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return
    
    num_dashes = max(1, int(dist / (dash_length * 2)))
    dx = (x2 - x1) / (num_dashes * 2)
    dy = (y2 - y1) / (num_dashes * 2)
    
    for i in range(num_dashes):
        start_x = int(x1 + i * 2 * dx)
        start_y = int(y1 + i * 2 * dy)
        end_x = int(x1 + (i * 2 + 1) * dx)
        end_y = int(y1 + (i * 2 + 1) * dy)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)

# Variables for movement smoothing
smooth_camera_rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
smooth_body_position = np.array([0.0, 0.0], dtype=np.float32)  # Body position (normalized X, Y)
smooth_depth_scale = 1.0  # Smoothed depth scale
smooth_size_factor = base_size_factor  # Smoothed rectangle size factor
smoothing_factor = 0.85  # Smoothing factor (0.0 = no smoothing, 1.0 = fully smooth) - Increased to reduce flickering
body_smoothing_factor = 0.90  # Higher smoothing for body (more stable)

# Variables to calculate FPS
fps_counter = 0
fps_start_time = time.time()
fps = 0.0

print("Starting...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame from camera!")
        break
    
    if frame is None or frame.size == 0:
        print("ERROR: Empty frame!")
        continue

    # Convert frame to mp.Image for face
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect face landmarks
    face_result = face_landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # Create full screen black canvas
    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Initialize size factor (default, no changes)
    # If no detection, use current smoothed value
    size_factor = smooth_size_factor
    
    # Variables for window effect (Johnny Chung Lee)
    head_pose_matrix = None
    face_success = False
    body_position = None  # Body position normalized (0.0 to 1.0)
    
    # Calculate body position based on face position
    # Use face position as reference to estimate body position
    if face_result.face_landmarks:
        face = face_result.face_landmarks[0]
        
        # Use face center (approximately the nose) as reference
        # The nose is approximately at the center of the head, which is at the center of the body
        nose_landmark = face[1]  # Landmark 1 is the nose
        
        # Body position is approximately the same as face in X,
        # but slightly lower in Y (body is below the head)
        body_center_x = nose_landmark.x
        body_center_y = nose_landmark.y + 0.1  # Adjust slightly downward for body
        
        # Normalize position (0.0 = left/top, 1.0 = right/bottom)
        body_position = np.array([body_center_x, body_center_y], dtype=np.float32)
        
        # Smooth body position
        smooth_body_position = smooth_body_position * body_smoothing_factor + body_position * (1 - body_smoothing_factor)
    
    # If face detected, use MediaPipe facial transformation matrix
    if face_result.face_landmarks and face_result.facial_transformation_matrixes:
        # Use facial transformation matrix (more accurate than solvePnP)
        # MediaPipe matrix is a list of 16 values (4x4 matrix in row-major format)
        matrix_data = face_result.facial_transformation_matrixes[0]
        head_pose_matrix = np.array(matrix_data).reshape(4, 4)
        face_success = True
        
        # Extract rotation and translation from 4x4 matrix
        # MediaPipe returns matrix in row-major format
        rotation_matrix = head_pose_matrix[:3, :3]
        translation = head_pose_matrix[:3, 3]
        
        # Convert rotation matrix to Euler angles (ZYX - Yaw, Pitch, Roll)
        # Use standard decomposition to get Euler angles
        sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        singular = sy < 1e-6
        
        if not singular:
            # Yaw (rotation around Y axis) - left/right
            yaw = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
            # Pitch (rotation around X axis) - up/down
            pitch = math.atan2(-rotation_matrix[2,0], sy)
            # Roll (rotation around Z axis) - tilt
            roll = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        else:
            yaw = math.atan2(-rotation_matrix[0,1], rotation_matrix[1,1])
            pitch = math.atan2(-rotation_matrix[2,0], sy)
            roll = 0
        
        # Calculate distance (Z from translation)
        # MediaPipe translation is normalized, need to scale it
        distance_z = abs(translation[2]) if len(translation) > 2 else 100.0
        # Scale distance to get more realistic values
        distance_z_scaled = distance_z * 100.0  # Approximate scale
        reference_distance = 100.0
        distance_factor = max(0.5, min(2.0, distance_z_scaled / reference_distance))
        
        # Rectangle size factor according to distance
        size_factor_target = base_size_factor / (1.0 + (distance_factor - 1.0) * 0.3)
        size_factor_target = np.clip(size_factor_target, 0.5, 0.9)
        
        # Smooth size factor
        smooth_size_factor = smooth_size_factor * smoothing_factor + size_factor_target * (1 - smoothing_factor)
        size_factor = smooth_size_factor
        
        # Scale according to distance
        scale = 1.0 + (distance_factor - 1.0) * 0.1
        scale = np.clip(scale, 0.8, 1.2)
        smooth_depth_scale = smooth_depth_scale * smoothing_factor + scale * (1 - smoothing_factor)
        
        # For window effect: when head moves, scene shifts in opposite direction
        # This creates the effect of looking through a fixed window
        target_rot = np.array([pitch, yaw, roll], dtype=np.float32)
        smooth_camera_rot = smooth_camera_rot * smoothing_factor + target_rot * (1 - smoothing_factor)
    
    # Calculate rectangle dimensions with current size factor
    # According to detailed video analysis, rectangle should be very rectangular (aspect ratio ~2.2)
    # Rectangle occupies almost the entire screen and is much wider than tall
    box_width = screen_width * size_factor
    box_height = screen_height * size_factor
    # Adjust to make it rectangular (aspect ratio ~2.2 as in video)
    aspect_target = 2.2
    current_aspect = box_width / box_height if box_height > 0 else 1.0
    if current_aspect < aspect_target:
        # If less rectangular than target, increase width
        box_width = box_height * aspect_target
    else:
        # If more rectangular, reduce height
        box_height = box_width / aspect_target
    
    # Base background rectangle (without additional transformation, as in box.py)
    base_rect_points = [
        [center_x - box_width/2, center_y - box_height/2],  # Top left
        [center_x + box_width/2, center_y - box_height/2],  # Top right
        [center_x + box_width/2, center_y + box_height/2],  # Bottom right
        [center_x - box_width/2, center_y + box_height/2]  # Bottom left
    ]
    
    # Initialize rectangle transformation
    # Rectangle moves according to person's position (inverse direction)
    rect_points_transformed = base_rect_points.copy()
    
    # If face or body detected, apply window effect (Johnny Chung Lee)
    # According to video: central rectangle is always visible but moves
    # When person is on the right, rectangle shifts left (and vice versa)
    if (face_success and head_pose_matrix is not None) or body_position is not None:
        # Rectangle displacement sensitivity
        # According to analysis: when person moves 6-25%, rectangle moves ~0.16-0.13x of that
        rectangle_sensitivity_x = 0.15  # Rectangle moves 15% of person's movement
        rectangle_sensitivity_y = 0.12  # Rectangle moves 12% of person's movement
        
        # Calculate rectangle displacement based on person's position
        offset_x = 0.0
        offset_y = 0.0
        
        # Use body position as main reference
        if body_position is not None:
            # When person is on the left (body_x < 0.5), rectangle shifts right
            # When person is on the right (body_x > 0.5), rectangle shifts left
            body_x_normalized = (smooth_body_position[0] - 0.5) * 2.0  # -1.0 to 1.0
            body_y_normalized = (smooth_body_position[1] - 0.5) * 2.0  # -1.0 to 1.0
            
            # Inverse displacement: person on left -> rectangle on right
            offset_x = -body_x_normalized * screen_width * rectangle_sensitivity_x
            offset_y = -body_y_normalized * screen_height * rectangle_sensitivity_y
        
        # Additional adjustment based on head rotation
        if face_success and head_pose_matrix is not None:
            yaw_degrees = smooth_camera_rot[1] * 180.0 / math.pi
            pitch_degrees = smooth_camera_rot[0] * 180.0 / math.pi
            
            # Subtle adjustment based on head rotation
            offset_x += -yaw_degrees * screen_width * rectangle_sensitivity_x * 0.3 / 90.0
            offset_y += -pitch_degrees * screen_height * rectangle_sensitivity_y * 0.3 / 90.0
        
        # Limit displacement to keep rectangle always visible
        # Rectangle must be able to move but always within screen
        max_offset_x = box_width * 0.3  # Allow displacement up to 30% of rectangle
        max_offset_y = box_height * 0.3
        offset_x = np.clip(offset_x, -max_offset_x, max_offset_x)
        offset_y = np.clip(offset_y, -max_offset_y, max_offset_y)
        
        # Apply displacement to rectangle
        # Rectangle changes position but always remains visible
        new_center_x = center_x + offset_x
        new_center_y = center_y + offset_y
        
        # Ensure complete rectangle is within screen
        margin_x = box_width / 2
        margin_y = box_height / 2
        new_center_x = np.clip(new_center_x, margin_x, screen_width - margin_x)
        new_center_y = np.clip(new_center_y, margin_y, screen_height - margin_y)
        
        # Transform each rectangle point with new center
        rect_points_transformed = []
        for pt in base_rect_points:
            # Relative displacement from original center
            dx = pt[0] - center_x
            dy = pt[1] - center_y
            
            # Apply scale according to distance (approach/move away)
            dx_scaled = dx * smooth_depth_scale
            dy_scaled = dy * smooth_depth_scale
            
            # Apply new center
            x_final = new_center_x + dx_scaled
            y_final = new_center_y + dy_scaled
            
            rect_points_transformed.append([x_final, y_final])
    
    # Build lines with transformed rectangle
    all_lines = build_lines_array(rect_points_transformed)
    
    # Draw all lines with increased thickness
    line_thickness = 3  # Line thickness (increased from 1 to 3)
    for line in all_lines:
        x1, y1, x2, y2 = line
        cv2.line(screen, (x1, y1), (x2, y2), (255, 255, 255), line_thickness, cv2.LINE_AA)
    
    # Show camera frame at bottom right
    # Flip horizontally (mirror) to correct left-right inversion
    frame_flipped = cv2.flip(frame, 1)  # flipCode=1 flips horizontally
    camera_frame_resized = cv2.resize(frame_flipped, (camera_window_width, camera_window_height))
    y_offset = screen_height - camera_window_height
    x_offset = screen_width - camera_window_width
    screen[y_offset:y_offset+camera_window_height, x_offset:x_offset+camera_window_width] = camera_frame_resized
    
    # Draw a frame around the camera window
    cv2.rectangle(screen, (x_offset-2, y_offset-2), 
                  (x_offset+camera_window_width+2, y_offset+camera_window_height+2), 
                  (255, 255, 255), 2)
    
    # Calculate FPS every second
    fps_counter += 1
    elapsed = time.time() - fps_start_time
    if elapsed >= 1.0:
        fps = fps_counter / elapsed
        fps_counter = 0
        fps_start_time = time.time()
    
    # Update window title with FPS and detection status
    body_status = "Body: OK" if body_position is not None else "Body: NO"
    face_status = "Face: OK" if (face_success and head_pose_matrix is not None) else "Face: NO"
    window_title = f"3D Window - Body & Head Tracking (Johnny Chung Lee 2008) - FPS: {fps:.1f} - {body_status} {face_status}"
    cv2.setWindowTitle(window_name, window_title)

    cv2.imshow(window_name, screen)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
