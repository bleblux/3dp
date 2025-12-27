# 3D Window - Perspective Correction with Head Tracking

A real-time 3D window effect application that creates an immersive perspective illusion based on Johnny Chung Lee's experiments (2008). The screen acts as a fixed window through which a 3D scene is viewed, and the perspective changes dynamically as you move your head and body.

## Demo Video

You can see the application in action by viewing the [demo video](demo.mp4) included in this repository.

## Features

- **Real-time head tracking** using MediaPipe Face Landmarker
- **Body position estimation** for enhanced perspective control
- **Dynamic 3D perspective** that responds to your movements
- **Smooth motion tracking** with configurable smoothing factors
- **FPS counter** and detection status display
- **Camera preview** window in the bottom right corner

## Requirements

- Python 3.8 or higher
- Webcam/camera connected to your computer
- MediaPipe Face Landmarker model file (`face_landmarker.task`)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd 3dp
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` (>=4.8.0) - For camera and image processing
- `numpy` (>=1.24.0) - For numerical computations
- `mediapipe` (>=0.10.0) - For face and pose tracking

### 3. Download the MediaPipe Face Landmarker model

The application requires the MediaPipe Face Landmarker model file. You need to download it manually:

1. Visit the [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) page
2. Download the face landmarker model file
3. Place the file named `face_landmarker.task` in the project root directory

Alternatively, you can download it directly using:
- **Direct download link**: [Face Landmarker Model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

**Important**: The model file must be named exactly `face_landmarker.task` and placed in the same directory as `3dp.py`.

## Usage

### Running the 3D Window application

```bash
python 3dp.py
```

The application will:
1. Open your default camera (camera ID 0)
2. Create a full-screen window showing the 3D perspective effect
3. Track your face and body movements in real-time
4. Display FPS and detection status in the window title

**Controls:**
- Press `ESC` to exit the application

### Running the Box 3D demo

For a simpler static 3D box visualization (without tracking):

```bash
python box.py
```

**Controls:**
- Press `ESC` to exit

## Project Structure

```
3dp/
├── 3dp.py                  # Main application with head/body tracking
├── box.py                   # Simple 3D box visualization demo
├── face_landmarker.task     # MediaPipe Face Landmarker model (download required)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Configuration

### Screen Resolution

By default, the application uses a resolution of 1920x1080. You can modify this in `3dp.py`:

```python
screen_width = 1920   # Adjust according to your screen
screen_height = 1080  # Adjust according to your screen
```

### Camera Settings

The application uses camera ID 0 by default. To use a different camera, modify:

```python
camera_id = 0  # Change to 1, 2, etc. for other cameras
```

### Smoothing Parameters

You can adjust the smoothing factors for more or less responsive tracking:

```python
smoothing_factor = 0.85        # General smoothing (0.0 = no smoothing, 1.0 = fully smooth)
body_smoothing_factor = 0.90  # Body position smoothing (higher = more stable)
```

### Rectangle Sensitivity

Adjust how much the 3D rectangle moves in response to your movements:

```python
rectangle_sensitivity_x = 0.15  # Horizontal sensitivity (15% of movement)
rectangle_sensitivity_y = 0.12  # Vertical sensitivity (12% of movement)
```

## Troubleshooting

### Camera not opening

- **Error**: `ERROR: Could not open camera 0!`
- **Solution**: 
  - Check that your camera is connected and not being used by another application
  - Try changing `camera_id` to 1 or 2 if you have multiple cameras
  - On Linux, you may need to install `v4l-utils` or grant camera permissions

### Model file not found

- **Error**: FileNotFoundError or similar when starting
- **Solution**: 
  - Ensure `face_landmarker.task` is in the same directory as `3dp.py`
  - Verify the file name is exactly `face_landmarker.task` (case-sensitive)
  - Re-download the model file if corrupted

### Low FPS or laggy performance

- **Solutions**:
  - Reduce screen resolution in the code
  - Close other applications using the camera
  - Lower the smoothing factors for more responsive (but potentially jittery) tracking
  - Ensure you have a good lighting environment for better face detection

### Face/body not detected

- **Solutions**:
  - Ensure good lighting conditions
  - Position yourself in front of the camera
  - Check that your face is clearly visible
  - The window title will show "Body: NO" or "Face: NO" if detection fails

### Import errors

- **Error**: `ModuleNotFoundError: No module named 'cv2'` or similar
- **Solution**: 
  ```bash
  pip install -r requirements.txt
  ```
  Or install packages individually:
  ```bash
  pip install opencv-python numpy mediapipe
  ```

## How It Works

The application implements a 3D window effect inspired by Johnny Chung Lee's research:

1. **Face Detection**: Uses MediaPipe Face Landmarker to detect facial landmarks and head pose
2. **Body Position Estimation**: Estimates body position based on face position (using nose landmark as reference)
3. **Perspective Calculation**: Calculates 3D perspective transformations based on head rotation (yaw, pitch, roll)
4. **Rectangle Transformation**: Moves and scales the central 3D rectangle inversely to your movements, creating the window effect
5. **Line Rendering**: Draws perspective lines that extend from the rectangle to screen edges

The effect creates an illusion that you're looking through a fixed window, with the scene moving in the opposite direction of your head movements.

## Author

**Josep Santos Pujol**

- LinkedIn: [https://www.linkedin.com/in/josepsantos/](https://www.linkedin.com/in/josepsantos/)

## Credits

- Based on research by **Johnny Chung Lee** (2008) on head tracking for 3D displays
- Uses **MediaPipe** by Google for face and pose tracking
- Built with **OpenCV** and **NumPy**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

MIT License is one of the most permissive open-source licenses, allowing:
- Commercial use
- Modification
- Distribution
- Private use
- Sublicensing

The only requirement is to include the original copyright notice and license text.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Studio Joanie Lemercier [https://www.facebook.com/studiojoanielemercier][https://www.linkedin.com/in/joanie-lemercier-98153757/]. Your idea was a true inspiration to me. Thank you for sharing it.
- Johnny Chung Lee for the original research on head tracking
- Google MediaPipe team for the excellent tracking library
- OpenCV community for computer vision tools

