import cv2
import numpy as np

def initialize_camera(camera_index=0):
    """Connect to the camera and return the video capture object."""
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None, None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        focal_length = frame_width  
        principal_point = (frame_width / 2, frame_height / 2)
        distortion_coefficients = np.zeros((4, 1))  # Assume no lens distortion

        print(f"Camera connected successfully!")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps}")

        return cap, {
            'frame_width': frame_width,
            'frame_height': frame_height,
            'fps': fps,
            'brightness': brightness,
            'exposure': exposure,
            'focal_length': focal_length,
            'principal_point': principal_point,
            'distortion_coefficients': distortion_coefficients
        }
    except Exception as e:
        print(f"Error connecting to camera: {e}")
        return None, None

def set_camera_property(cap, prop_id, value):
    """Set a specific camera property."""
    if cap is None:
        print("Camera not connected")
        return False
    cap.set(prop_id, value)
    return True

def get_camera_property(cap, prop_id):
    """Get a specific camera property."""
    if cap is None:
        print("Camera not connected")
        return None
    return cap.get(prop_id)

def set_camera_properties(cap, brightness=None, exposure=None):
    """Modify camera properties programmatically"""
    if cap is None:
        print("Camera not connected")
        return False

    if brightness is not None:
        if set_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, brightness):
            print(f"Brightness set to: {brightness}")

    if exposure is not None:
        if set_camera_property(cap, cv2.CAP_PROP_EXPOSURE, exposure):
            print(f"Exposure set to: {exposure}")

    return True

def get_intrinsic_parameters(camera_info):
    """Return the intrinsic parameters."""
    if camera_info:
        return camera_info['focal_length'], camera_info['principal_point'], camera_info['distortion_coefficients']
    return None, None, None

def set_intrinsic_parameters(camera_info, focal_length=None, principal_point=None, distortion_coefficients=None):
    """Programmatically modify intrinsic parameters (simulation)."""
    print("\nAttempting to modify intrinsic parameters programmatically (simulation):")
    if camera_info is None:
        print("Camera information not initialized.")
        return camera_info

    updated_info = camera_info.copy()

    if focal_length is not None:
        updated_info['focal_length'] = focal_length
        print(f"Focal Length set to: {focal_length:.1f} (simulated)")
    if principal_point is not None and isinstance(principal_point, tuple) and len(principal_point) == 2:
        updated_info['principal_point'] = principal_point
        print(f"Principal Point set to: {principal_point} (simulated)")
    # if distortion_coefficients is not None and isinstance(distortion_coefficients, np.ndarray) and distortion_coefficients.shape == (4, 1):
    #     updated_info['distortion_coefficients'] = distortion_coefficients.astype(np.float64)
    #     print(f"Distortion Coefficients set to:\n{updated_info['distortion_coefficients'].flatten()} (simulated)")
    if distortion_coefficients is not None and isinstance(distortion_coefficients, np.ndarray) and distortion_coefficients.shape == (4, 1):
        updated_info['distortion_coefficients'] = distortion_coefficients.astype(np.float64)
        formatted_coeffs = ' '.join([str(int(x)) for x in updated_info['distortion_coefficients'].flatten()])
        print(f"Distortion Coefficients set to:\n[{formatted_coeffs}] (simulated)")
    else:
        print("Invalid format for distortion coefficients, keeping estimated values.")

    print("Updated Intrinsic Parameters (Simulated):")
    focal, pp, dist = get_intrinsic_parameters(updated_info)
    if focal is not None and pp is not None and dist is not None:
        print(f"Camera Matrix:\n[[{focal:.1f}, 0.0, {pp[0]:.1f}],\n [0.0, {focal:.1f}, {pp[1]:.1f}],\n [0.0, 0.0, 1.0]]")
        print("Distortion Coefficients:")
        print('[' + ' '.join([str(int(x)) for x in dist.flatten()]) + ']')
        #print(f"Distortion Coefficients:\n{dist.flatten()}")
    return updated_info

def capture_frame(cap):
    """Capture a single frame from the camera."""
    if cap is None:
        print("Camera not connected.")
        return None, False
    ret, frame = cap.read()
    return frame, ret

def release_camera(cap):
    """Release the camera and destroy all windows."""
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

def display_camera_properties(cap, camera_info, detector_classical=None):
    """Display current camera properties and detector parameters."""
    if cap is None:
        print("Camera not connected.")
        return

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

    print("\nCurrent Camera Properties:")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Brightness: {brightness}")
    print(f"Exposure: {exposure}")

    if camera_info:
        focal_length = camera_info.get('focal_length')
        principal_point = camera_info.get('principal_point')
        distortion_coefficients = camera_info.get('distortion_coefficients')

        print("\nIntrinsic Parameters:")
        print(f"Focal Length: {focal_length:.1f}")
        print(f"Principal Point: {principal_point}")
        # print(f"Distortion Coefficients:\n{distortion_coefficients.flatten()}")
        print("Distortion Coefficients:")
        print('[' + ' '.join([str(int(x)) for x in distortion_coefficients.flatten()]) + ']')

    if detector_classical:
        hsv_lower, hsv_upper = detector_classical.get_hsv_range()
        area_lower, area_upper = detector_classical.get_area_range()
        circularity_threshold = detector_classical.get_circularity_threshold()

        print("\nClassical CV Detector Parameters:")
        print(f"Area Range: {area_lower} - {area_upper}")
        print(f"Circularity Threshold: {circularity_threshold:.2f}")
        print(f"HSV Range: {hsv_lower.tolist()} - {hsv_upper.tolist()}")