import cv2
from ultralytics import YOLO

# --- Configuration ---

# The path to your ONNX model file (e.g., exported from a YOLOv8 model)
ONNX_MODEL_PATH = r'ONNX model here'

# The path to the BoT-SORT configuration YAML file from Ultralytics
# You can find this in the Ultralytics repository, e.g., 'ultralytics/cfg/trackers/botsort.yaml'
TRACKER_CONFIG_PATH = "botsort.yaml"

# Your custom class map: {model_class_id: "Custom_Name"}
# NOTE: The keys (0, 2, 16, 80, 81) must be the class IDs *outputted by your ONNX model*.
CUSTOM_CLASS_NAMES = {
    0: "Person",
    2: "Car",
    16: "Dog",
    80: "Box",
    81: "Redball"
}

# Get a list of class IDs to filter for tracking (i.e., the keys of your map)
# This is used by the 'classes' argument in the track function.
FILTER_CLASSES = list(CUSTOM_CLASS_NAMES.keys())

# --- Tracking and Webcam Setup ---

def run_tracking_webcam():
    # Load the ONNX model using the Ultralytics YOLO class
    try:
        model = YOLO(ONNX_MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your ONNX_MODEL_PATH and ensure the model is a valid detection model.")
        return

    # Initialize webcam capture (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam tracking. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # 1. Mirror the video frame (flip horizontally)
            mirrored_frame = cv2.flip(frame, 1)

            # 2. Run object tracking
            # - source: The current frame (NumPy array)
            # - tracker: Path to the tracker configuration file (botsort.yaml)
            # - classes: Filters predictions to only include the specified class IDs
            # - persist: Keeps the tracker state between frames
            # - verbose: Set to False to prevent excessive console output
            results = model.track(
                source=mirrored_frame, 
                tracker=TRACKER_CONFIG_PATH, 
                classes=FILTER_CLASSES, 
                persist=True, 
                verbose=False,
                device=0, # Use GPU (set to 'cpu' if no CUDA-enabled GPU is available)
                conf=0.55,
                iou=0.45
                
            )

            # 3. Get the annotated frame from the results object
            # The .plot() method automatically draws bounding boxes, labels, and track IDs.
            annotated_frame = results[0].plot()

            # The class labels on the plot will still use the model's original names 
            # (e.g., 'person', 'car', etc.) unless you use the 'names' argument in the track call.
            # However, since you are filtering by class ID, only your desired objects will be tracked.
            
            # For a cleaner display that shows *only* your custom names, 
            # you can modify the model's internal names dictionary:
            model.names.update(CUSTOM_CLASS_NAMES)
            # Re-run tracking for the next frame with updated names.

            # 4. Display the resulting frame
            cv2.imshow("Webcam Live detection", annotated_frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # If no frame is read, break the loop
            print("Webcam feed lost.")
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking session ended.")

if __name__ == "__main__":
    run_tracking_webcam()
