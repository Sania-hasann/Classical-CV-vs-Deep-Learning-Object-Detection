import cv2
import numpy as np
import camera_utils
import classical_cv
import deep_learning_detection

def main():
    cap, camera_info = camera_utils.initialize_camera()  # Initialize camera once

    if cap is None:
        print("Failed to initialize camera, exiting.")
        return

    # Create an instance of the CircularObjectDetector class
    detector_classical = classical_cv.CircularObjectDetector(camera_info)
    detector_classical.set_hsv_range([0, 50, 50], [180, 255, 255])
    detector_classical.set_area_range(500, 50000)
    detector_classical.set_circularity_threshold(0.7)

    camera_utils.set_camera_properties(cap, brightness=98) 
    camera_utils.set_camera_properties(cap, exposure=-3)  
    camera_utils.set_intrinsic_parameters(camera_info, focal_length=600.0, principal_point=(330, 250), distortion_coefficients=np.zeros((4, 1)))

    while True:
        print("\nChoose an option:")
        print("1. Classical Computer Vision Object Detection")
        print("2. Deep Learning Object Detection")
        print("3. Display Camera Properties (Part 1)")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1': # Part 2 - Classical CV
            detector_classical.run(cap)  
        elif choice == '2': # Part 3 - Deep Learning
            deep_learning_detection.run_object_detection_dl(cap, camera_info) 
        elif choice == '3': # Part 1 - Image Acquisition & Camera Properties
            camera_utils.display_camera_properties(cap, camera_info, detector_classical)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

    camera_utils.release_camera(cap)

if __name__ == "__main__":
    main()

