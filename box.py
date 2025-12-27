import cv2
import numpy as np
import time

# Screen dimensions
screen_width = 1920
screen_height = 1080

# Create main window
cv2.namedWindow("Box 3D", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Box 3D", screen_width, screen_height)
cv2.moveWindow("Box 3D", 0, 0)

# Rectangle dimensions (20% smaller than window)
# If window is 100%, rectangle is 80%
box_width = screen_width * 0.7
box_height = screen_height * 0.7

# Screen center
center_x = screen_width / 2
center_y = screen_height / 2

# Create the 4 rectangle points (2D to start)
# Rectangle centered on screen
rect_points = [
    [center_x - box_width/2, center_y - box_height/2],  # Top left
    [center_x + box_width/2, center_y - box_height/2],  # Top right
    [center_x + box_width/2, center_y + box_height/2],  # Bottom right
    [center_x - box_width/2, center_y + box_height/2]  # Bottom left
]

def normalize_line(x1, y1, x2, y2):
    """Normalizes a line to avoid duplicates (smallest point first)"""
    if (x1, y1) < (x2, y2):
        return (int(x1), int(y1), int(x2), int(y2))
    else:
        return (int(x2), int(y2), int(x1), int(y1))

def build_lines_array():
    """Builds an array with all unique lines to draw"""
    lines_set = set()
    
    # 1. Background rectangle outline
    for i in range(4):
        pt1 = rect_points[i]
        pt2 = rect_points[(i + 1) % 4]
        line = normalize_line(pt1[0], pt1[1], pt2[0], pt2[1])
        lines_set.add(line)
    
    # 2. Calculate edge inclinations for vertical lines
    left_top_dx = 0 - rect_points[0][0]
    left_top_dy = 0 - rect_points[0][1]
    left_bottom_dx = 0 - rect_points[3][0]
    left_bottom_dy = screen_height - rect_points[3][1]
    
    right_top_dx = screen_width - rect_points[1][0]
    right_top_dy = 0 - rect_points[1][1]
    right_bottom_dx = screen_width - rect_points[2][0]
    right_bottom_dy = screen_height - rect_points[2][1]
    
    # 3. Vertical lines of the background rectangle and their extensions
    for i in range(1, 8):  # 7 vertical lines
        x = center_x - box_width/2 + (i * box_width / 8)
        y_top = center_y - box_height/2
        y_bottom = center_y + box_height/2
        
        # Vertical line inside the rectangle
        line = normalize_line(x, y_top, x, y_bottom)
        lines_set.add(line)
        
        # Calculate relative position within rectangle (0.0 = left, 1.0 = right)
        t = (i - 1) / 7.0 if 7 > 0 else 0
        
        # Interpolate inclination between left and right for top part
        top_dx = left_top_dx + t * (right_top_dx - left_top_dx)
        top_dy = left_top_dy + t * (right_top_dy - left_top_dy)
        if top_dy != 0:
            top_factor = -y_top / top_dy
            top_x_end = x + top_dx * top_factor
        else:
            top_x_end = x
        
        # Interpolate inclination between left and right for bottom part
        bottom_dx = left_bottom_dx + t * (right_bottom_dx - left_bottom_dx)
        bottom_dy = left_bottom_dy + t * (right_bottom_dy - left_bottom_dy)
        if bottom_dy != 0:
            bottom_factor = (screen_height - y_bottom) / bottom_dy
            bottom_x_end = x + bottom_dx * bottom_factor
        else:
            bottom_x_end = x
        
        # Top extension
        line = normalize_line(x, y_top, top_x_end, 0)
        lines_set.add(line)
        
        # Bottom extension
        line = normalize_line(x, y_bottom, bottom_x_end, screen_height)
        lines_set.add(line)
    
    # 4. Calculate edge inclinations for horizontal lines
    top_left_dx = 0 - rect_points[0][0]
    top_left_dy = 0 - rect_points[0][1]
    top_right_dx = screen_width - rect_points[1][0]
    top_right_dy = 0 - rect_points[1][1]
    
    bottom_left_dx = 0 - rect_points[3][0]
    bottom_left_dy = screen_height - rect_points[3][1]
    bottom_right_dx = screen_width - rect_points[2][0]
    bottom_right_dy = screen_height - rect_points[2][1]
    
    # 5. Horizontal lines of the background rectangle and their extensions
    for j in range(1, 8):  # 7 horizontal lines
        y = center_y - box_height/2 + (j * box_height / 8)
        x_left = center_x - box_width/2
        x_right = center_x + box_width/2
        
        # Horizontal line inside the rectangle
        line = normalize_line(x_left, y, x_right, y)
        lines_set.add(line)
        
        # Calculate relative position within rectangle (0.0 = top, 1.0 = bottom)
        t = (j - 1) / 7.0 if 7 > 0 else 0
        
        # Interpolate inclination between top and bottom for left part
        left_dx = top_left_dx + t * (bottom_left_dx - top_left_dx)
        left_dy = top_left_dy + t * (bottom_left_dy - top_left_dy)
        if left_dx != 0:
            left_factor = -x_left / left_dx
            left_y_end = y + left_dy * left_factor
        else:
            left_y_end = y
        
        # Interpolate inclination between top and bottom for right part
        right_dx = top_right_dx + t * (bottom_right_dx - top_right_dx)
        right_dy = top_right_dy + t * (bottom_right_dy - top_right_dy)
        if right_dx != 0:
            right_factor = (screen_width - x_right) / right_dx
            right_y_end = y + right_dy * right_factor
        else:
            right_y_end = y
        
        # Left extension
        line = normalize_line(x_left, y, 0, left_y_end)
        lines_set.add(line)
        
        # Right extension
        line = normalize_line(x_right, y, screen_width, right_y_end)
        lines_set.add(line)
    
    # 6. 4 lines from rectangle edges (3D rectangle edges)
    # Top left: rectangle → (0, 0)
    line = normalize_line(rect_points[0][0], rect_points[0][1], 0, 0)
    lines_set.add(line)
    
    # Top right: rectangle → (screen_width, 0)
    line = normalize_line(rect_points[1][0], rect_points[1][1], screen_width, 0)
    lines_set.add(line)
    
    # Bottom right: rectangle → (screen_width, screen_height)
    line = normalize_line(rect_points[2][0], rect_points[2][1], screen_width, screen_height)
    lines_set.add(line)
    
    # Bottom left: rectangle → (0, screen_height)
    line = normalize_line(rect_points[3][0], rect_points[3][1], 0, screen_height)
    lines_set.add(line)
    
    # 7. Intermediate rectangles (outlines only)
    for layer in range(1, 8):  # 7 additional rectangles
        t = layer / 8.0
        
        layer_width = box_width + t * (screen_width - box_width)
        layer_height = box_height + t * (screen_height - box_height)
        
        layer_center_x = center_x
        layer_center_y = center_y
        
        layer_points = [
            [layer_center_x - layer_width/2, layer_center_y - layer_height/2],
            [layer_center_x + layer_width/2, layer_center_y - layer_height/2],
            [layer_center_x + layer_width/2, layer_center_y + layer_height/2],
            [layer_center_x - layer_width/2, layer_center_y + layer_height/2]
        ]
        
        # Outline of this layer's rectangle
        for i in range(4):
            pt1 = layer_points[i]
            pt2 = layer_points[(i + 1) % 4]
            line = normalize_line(pt1[0], pt1[1], pt2[0], pt2[1])
            lines_set.add(line)
    
    return list(lines_set)

# Build lines array once (outside the loop)
all_lines = build_lines_array()

print(f"Total unique lines: {len(all_lines)}")
print("Starting Box 3D...")
print("Press ESC to exit")

# Variables to calculate FPS
fps_counter = 0
fps_start_time = time.time()
fps = 0.0

while True:
    # Start timer for this frame
    frame_start = time.time()
    
    # Black canvas
    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Draw all lines from array
    for line in all_lines:
        x1, y1, x2, y2 = line
        cv2.line(screen, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    
    # Calculate FPS every second
    fps_counter += 1
    elapsed = time.time() - fps_start_time
    if elapsed >= 1.0:
        fps = fps_counter / elapsed
        fps_counter = 0
        fps_start_time = time.time()
    
    # Update window title with FPS and line count
    window_title = f"Box 3D - FPS: {fps:.1f} - Lines: {len(all_lines)}"
    cv2.setWindowTitle("Box 3D", window_title)
    
    cv2.imshow("Box 3D", screen)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
