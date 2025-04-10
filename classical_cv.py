# import cv2
# import numpy as np
# import time
# from camera_utils import capture_frame

# class CircularObjectDetector:
#     def __init__(self, camera_info=None):
#         self.camera_info = camera_info

#         # Detection parameters 
#         self.hsv_lower = np.array([0, 0, 0]) 
#         self.hsv_upper = np.array([180, 255, 255]) 
#         self.min_area = 300  
#         self.max_area = 100000 
#         self.circularity_threshold = 0.7  

#     def set_hsv_range(self, lower, upper):
#         """Set HSV color range for filtering"""
#         self.hsv_lower = np.array(lower)
#         self.hsv_upper = np.array(upper)
#         print(f"HSV range set to: {lower} - {upper}")
#         return self.hsv_lower, self.hsv_upper

#     def get_hsv_range(self):
#         """Get current HSV color range"""
#         return self.hsv_lower, self.hsv_upper

#     def set_area_range(self, min_area, max_area):
#         """Set min and max area for contour filtering"""
#         self.min_area = min_area
#         self.max_area = max_area
#         print(f"Area range set to: {min_area} - {max_area}")
#         return self.min_area, self.max_area

#     def get_area_range(self):
#         """Get current area range"""
#         return self.min_area, self.max_area

#     def set_circularity_threshold(self, threshold):
#         """Set the circularity threshold"""
#         self.circularity_threshold = threshold
#         print(f"Circularity threshold set to: {threshold}")
#         return self.circularity_threshold

#     def get_circularity_threshold(self):
#         """Get current circularity threshold"""
#         return self.circularity_threshold

#     def color_filtering(self, frame):
#         """Filter the frame based on HSV color range"""
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

#         return mask

#     def morphological_operations(self, mask):
#         """Apply morphological operations to remove noise"""
#         kernel = np.ones((5, 5), np.uint8)
#         eroded = cv2.erode(mask, kernel, iterations=1)
#         dilated = cv2.dilate(eroded, kernel, iterations=1)
#         closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

#         return closed

#     def edge_detection(self, processed_mask):
#         """Apply Canny edge detection"""
#         blurred = cv2.GaussianBlur(processed_mask, (5, 5), 0)
#         edges = cv2.Canny(blurred, 30, 150)

#         return edges

#     def contour_analysis(self, edges, original_frame):
#         """Detect and analyze contours to find circular objects"""
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         result_frame = original_frame.copy()
#         circular_objects = []

#         for contour in contours:
#             area = cv2.contourArea(contour)

#             if self.min_area <= area <= self.max_area:
#                 perimeter = cv2.arcLength(contour, True)
#                 approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

#                 #circularity = (4π × Area / Perimeter²)
#                 circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

#                 # Check if the shape is circular
#                 if circularity >= self.circularity_threshold:
#                     # Get bounding rectangle
#                     x, y, w, h = cv2.boundingRect(contour)

#                     # Draw bounding box (rectangle)
#                     cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                     # Fit a circle to the contour
#                     (c_x, c_y), radius = cv2.minEnclosingCircle(contour)
#                     center = (int(c_x), int(c_y))
#                     radius = int(radius)

#                     cv2.circle(result_frame, center, radius, (0, 165, 255), 2)
#                     cv2.putText(result_frame, f"Circ: {circularity:.2f}", (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                     circular_objects.append({
#                         'contour': contour,
#                         'center': center,
#                         'radius': radius,
#                         'circularity': circularity,
#                         'area': area
#                     })

#         return result_frame, circular_objects

#     def process_frame(self, frame):
#         """Process a single frame through the entire pipeline"""
#         mask = self.color_filtering(frame)
#         processed_mask = self.morphological_operations(mask)
#         edges = self.edge_detection(processed_mask)
#         result_frame, detected_objects = self.contour_analysis(edges, frame)

#         h, w = frame.shape[:2]
#         mask_resized = cv2.resize(mask, (w // 3, h // 3))
#         processed_mask_resized = cv2.resize(processed_mask, (w // 3, h // 3))
#         edges_resized = cv2.resize(edges, (w // 3, h // 3))

#         mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
#         processed_mask_color = cv2.cvtColor(processed_mask_resized, cv2.COLOR_GRAY2BGR)
#         edges_color = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)

#         intermediate_row = np.hstack([mask_color, processed_mask_color, edges_color])

#         cv2.putText(intermediate_row, "HSV Mask", (10, 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         cv2.putText(intermediate_row, "Morphological Ops", (w // 3 + 10, 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         cv2.putText(intermediate_row, "Edge Detection", (2 * w // 3 + 10, 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         cv2.putText(result_frame, f"Detected: {len(detected_objects)}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         intermediate_height = intermediate_row.shape[0]
#         result_with_steps = np.vstack([result_frame, np.zeros((intermediate_height, w, 3), dtype=np.uint8)])
#         result_with_steps[result_frame.shape[0]:, :intermediate_row.shape[1]] = intermediate_row

#         return result_with_steps, detected_objects

#     def run(self, cap):
#         """Run the detector on live camera feed"""
#         if cap is None or not cap.isOpened():
#             print("Error: Camera not connected")
#             return

#         cv2.namedWindow("Classical CV Object Detection", cv2.WINDOW_NORMAL)

#         try:
#             while True:
#                 frame, ret = capture_frame(cap)
#                 if not ret:
#                     print("Error: Failed to capture frame")
#                     break

#                 start_time = time.time()
#                 result_frame, detected_objects = self.process_frame(frame)
#                 end_time = time.time()

#                 time_difference = end_time - start_time
#                 if time_difference > 0:
#                     fps = 1 / time_difference
#                     cv2.putText(result_frame, f"FPS: {fps:.2f}", (result_frame.shape[1] - 150, 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 else:
#                     cv2.putText(result_frame, "FPS: N/A", (result_frame.shape[1] - 150, 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#                 cv2.imshow("Classical CV Object Detection", result_frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#         finally:
#             cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import os
from camera_utils import capture_frame

class CircularObjectDetector:
    def __init__(self, camera_info=None):
        self.camera_info = camera_info

        # Detection parameters
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([180, 255, 255])
        self.min_area = 300
        self.max_area = 100000
        self.circularity_threshold = 0.7
        self.screenshot_folder = "screenshots"
        os.makedirs(self.screenshot_folder, exist_ok=True)
        self.screenshot_counter = 0
        self.last_screenshot_time = time.time()
        self.screenshot_interval = 5  # seconds

    def set_hsv_range(self, lower, upper):
        """Set HSV color range for filtering"""
        self.hsv_lower = np.array(lower)
        self.hsv_upper = np.array(upper)
        print(f"HSV range set to: {lower} - {upper}")
        return self.hsv_lower, self.hsv_upper

    def get_hsv_range(self):
        """Get current HSV color range"""
        return self.hsv_lower, self.hsv_upper

    def set_area_range(self, min_area, max_area):
        """Set min and max area for contour filtering"""
        self.min_area = min_area
        self.max_area = max_area
        print(f"Area range set to: {min_area} - {max_area}")
        return self.min_area, self.max_area

    def get_area_range(self):
        """Get current area range"""
        return self.min_area, self.max_area

    def set_circularity_threshold(self, threshold):
        """Set the circularity threshold"""
        self.circularity_threshold = threshold
        print(f"Circularity threshold set to: {threshold}")
        return self.circularity_threshold

    def get_circularity_threshold(self):
        """Get current circularity threshold"""
        return self.circularity_threshold

    def color_filtering(self, frame):
        """Filter the frame based on HSV color range"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        return mask

    def morphological_operations(self, mask):
        """Apply morphological operations to remove noise"""
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        return closed

    def edge_detection(self, processed_mask):
        """Apply Canny edge detection"""
        blurred = cv2.GaussianBlur(processed_mask, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)

        return edges

    def contour_analysis(self, edges, original_frame):
        """Detect and analyze contours to find circular objects"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_frame = original_frame.copy()
        circular_objects = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_area <= area <= self.max_area:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                #circularity = (4π × Area / Perimeter²)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Check if the shape is circular
                if circularity >= self.circularity_threshold:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    # Draw bounding box (rectangle) on the original frame
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Fit a circle to the contour (for visualization on result_frame)
                    (c_x, c_y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(c_x), int(c_y))
                    radius = int(radius)

                    cv2.circle(result_frame, center, radius, (0, 165, 255), 2)
                    cv2.putText(result_frame, f"Circ: {circularity:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    circular_objects.append({
                        'contour': contour,
                        'center': center,
                        'radius': radius,
                        'circularity': circularity,
                        'area': area
                    })

        return result_frame, circular_objects

    def process_frame(self, frame):
        """Process a single frame through the entire pipeline"""
        mask = self.color_filtering(frame)
        processed_mask = self.morphological_operations(mask)
        edges = self.edge_detection(processed_mask)
        result_frame, detected_objects = self.contour_analysis(edges, frame) # Pass original frame for drawing

        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask, (w // 3, h // 3))
        processed_mask_resized = cv2.resize(processed_mask, (w // 3, h // 3))
        edges_resized = cv2.resize(edges, (w // 3, h // 3))

        mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        processed_mask_color = cv2.cvtColor(processed_mask_resized, cv2.COLOR_GRAY2BGR)
        edges_color = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)

        intermediate_row = np.hstack([mask_color, processed_mask_color, edges_color])

        cv2.putText(intermediate_row, "HSV Mask", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(intermediate_row, "Morphological Ops", (w // 3 + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(intermediate_row, "Edge Detection", (2 * w // 3 + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(result_frame, f"Detected: {len(detected_objects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        intermediate_height = intermediate_row.shape[0]
        result_with_steps = np.vstack([result_frame, np.zeros((intermediate_height, w, 3), dtype=np.uint8)])
        result_with_steps[result_frame.shape[0]:, :intermediate_row.shape[1]] = intermediate_row

        return result_with_steps, detected_objects

    def run(self, cap):
        """Run the detector on live camera feed"""
        if cap is None or not cap.isOpened():
            print("Error: Camera not connected")
            return

        cv2.namedWindow("Classical CV Object Detection", cv2.WINDOW_NORMAL)

        try:
            while True:
                frame, ret = capture_frame(cap)
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                # Create a copy of the original frame to draw bounding boxes on for saving
                frame_with_boxes = frame.copy()

                start_time = time.time()
                result_frame, detected_objects = self.process_frame(frame_with_boxes) # Process the copy
                end_time = time.time()

                time_difference = end_time - start_time
                if time_difference > 0:
                    fps = 1 / time_difference
                    cv2.putText(result_frame, f"FPS: {fps:.2f}", (result_frame.shape[1] - 150, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(result_frame, "FPS: N/A", (result_frame.shape[1] - 150, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Classical CV Object Detection", result_frame)

                current_time = time.time()
                if current_time - self.last_screenshot_time >= self.screenshot_interval:
                    if detected_objects:
                        self.screenshot_counter += 1
                        screenshot_name = os.path.join(self.screenshot_folder, f"cv_{self.screenshot_counter}.png")
                        cv2.imwrite(screenshot_name, frame_with_boxes) # Save the frame with bounding boxes
                        print(f"Screenshot saved: {screenshot_name} (Detected {len(detected_objects)} circles)")
                        self.last_screenshot_time = current_time
                    else:
                        # Optionally, you can choose to save a screenshot even if no circles are detected
                        # self.screenshot_counter += 1
                        # screenshot_name = os.path.join(self.screenshot_folder, f"cv_{self.screenshot_counter}_no_circle.png")
                        # cv2.imwrite(screenshot_name, frame)
                        # print(f"Screenshot saved: {screenshot_name} (No circles detected)")
                        self.last_screenshot_time = current_time

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize camera (you might need to adjust the camera index)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    detector = CircularObjectDetector()
    detector.run(cap)

    cap.release()