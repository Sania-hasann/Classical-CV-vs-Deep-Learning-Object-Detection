# Computer Vision Object Detection System

## Project Overview
This repository contains a comprehensive computer vision system that implements both classical and deep learning-based approaches for real-time object detection using a live camera feed. The project was developed as part of a Computer Vision Engineer assessment task to demonstrate expertise in object detection techniques and understanding of camera properties.

## Objectives
- Develop a complete vision pipeline that processes live camera feeds
- Extract and modify camera properties including intrinsic parameters
- Implement object detection using classical computer vision methods
- Implement object detection using deep learning-based approaches
- Analyze performance differences between approaches and camera configurations

## Key Features
- Real-time video capture and processing
- Classical computer vision pipeline for circular shape detection
- Deep learning-based object detection using YOLOv11s
- Camera property extraction and modification
- Performance analysis under different conditions

## Project Structure
```
code/
│
├── model/
│ ├── yolo11s.pt
├── screenshots/
├── main.py                      # Entry point
├── camera_utils.py              # Camera handling functions (Part 1)
├── classical_cv.py              # Classical CV implementation (Part 2)
├── deep_learning_detection.py   # Deep learning implementation (Part 3)
└── requirements.txt
```

## Implementation Details

### 1. Image Acquisition & Camera Properties
- Real-time video capture using OpenCV (`cv2.VideoCapture(0)`)
- Extraction of camera properties (resolution, frame rate)
- Programmatic adjustment of brightness (95) and exposure (-2)
- Estimation and simulation of intrinsic camera parameters:
  - Focal length initially set to frame width, then modified to 600.0
  - Principal point initially set to center, then modified to (330, 250)
  - Zero distortion coefficient

### 2. Classical Computer Vision Approach
The classical pipeline implements circular object detection through:
- HSV color filtering with range [0,50,50] to [180,255,255]
- Morphological operations (erosion, dilation, closing)
- Canny edge detection
- Contour analysis with:
  - Area-based filtering (500-50000 pixels)
  - Circularity calculation (4π×Area/Perimeter²)
  - Circularity threshold of 0.7
- Drawing bounding boxes and enclosing circles with circularity scores

### 3. Deep Learning Approach
The deep learning implementation uses:
- Pre-trained YOLOv11s model from the Ultralytics library
- Class filtering for common objects:
  ```python
  ['person', 'book', 'scissors', 'teddy bear', 'hair drier', 
   'remote', 'cell phone', 'bottle', 'cup', 'spoon']
  ```
- Confidence threshold of 0.7
- Bounding box and label annotations
- Automatic screenshot capture when specific object combinations are detected

## Key Findings

### Performance Comparison
1. **Classical CV Approach**:
   - Successfully detected circular objects based on color, shape, and size
   - Highly sensitive to lighting conditions and parameter tuning
   - Limited to detecting only circular shapes
   - Required extensive parameter tuning for optimal performance
   - Significant impact from lighting variations and shadows

2. **Deep Learning Approach (YOLOv11)**:
   - Significantly outperformed classical methods in versatility
   - Detected a wide variety of object classes
   - Demonstrated greater robustness to environmental changes
   - Maintained consistent performance across varying lighting conditions
   - Higher accuracy in complex detection scenarios

### Camera Property Impact
- **Brightness and Exposure**: 
  - Classical methods were highly sensitive to these parameters
  - Lower brightness reduced effectiveness of color-based filtering
  - Overexposure washed out features needed for edge detection
  - Deep learning methods showed greater invariance to these changes

- **Intrinsic Parameters**: 
  - While the code simulated changes to focal length and principal point, the full integration into detection pipelines requires further development
  - Proper camera calibration would improve accuracy of size and position measurements

## Limitations and Future Work
- Lack of proper camera calibration limited the accuracy of size and position measurements
- Potential improvements:
  - Implement proper camera calibration using checkerboards
  - Explore advanced classical feature detectors
  - Fine-tune deep learning models on custom datasets
  - Implement tracking algorithms
  - Develop comprehensive evaluation metrics using annotated datasets
  - Create a systematic approach to analyze camera parameter impacts
  - Build a more user-friendly interface with robust error handling

## Requirements
See `requirements.txt` for a list of dependencies.

## Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

## Screenshots
The `screenshots` directory contains captured images of detected objects from the camera feed.
