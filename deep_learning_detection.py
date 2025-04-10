import cv2
import os
import time
from ultralytics import YOLO

model = YOLO("model\yolo11s.pt")

classes_to_detect = ['person', 'book', 'scissors', 'teddy bear', 'hair drier', 'remote' , 'cell phone' , 'bottle' ,'cup', 'spoon'] 
confidence_threshold = 0.7
screenshots_dir = "screenshots"
os.makedirs(screenshots_dir, exist_ok=True)
last_capture_time = {}
capture_interval = 5  # seconds

def run_object_detection_dl(cap, camera_info=None):
    """Runs deep learning based object detection on video frames, displays camera properties, and captures screenshots with combined object names."""
    if cap is None or not cap.isOpened() or model is None:
        print("Error: Camera not initialized or YOLO model not loaded.")
        return

    class_names = model.names

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        current_time = time.time()
        detected_classes_in_frame = set()
        detected_objects_with_confidence = []
        results = model(frame)

        for result in results: 
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]

                    if class_name in classes_to_detect and confidence > confidence_threshold:
                        detected_classes_in_frame.add(class_name)

                        # Draw bounding box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Add label
                        label = f'{class_name}: {confidence:.2f}'
                        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if detected_classes_in_frame:
            sorted_detected_classes = sorted(list(detected_classes_in_frame))
            screenshot_name_base = "+".join(sorted_detected_classes)
            screenshot_filename = os.path.join(screenshots_dir, f"{screenshot_name_base}_{int(current_time)}.png")

            detected_combination = tuple(sorted_detected_classes)

            if detected_combination not in last_capture_time or (current_time - last_capture_time[detected_combination]) >= capture_interval:
                cv2.imwrite(screenshot_filename, frame)
                print(f"Screenshot saved: {screenshot_filename}")
                last_capture_time[detected_combination] = current_time

        cv2.imshow('Deep Learning Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()