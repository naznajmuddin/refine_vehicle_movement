import cv2
import os
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
CLASS_NAMES_DICT = model.model.names

# Define selected class names for tracking
SELECTED_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# Mapping for custom class labels
CUSTOM_LABELS = {
    "car": "mobil",
    "motorcycle": "motor",
    "bus": "bus",
    "truck": "truck",
}


def process_webcam():
    # Open webcam (use 0 for the default webcam, or the index of your camera)
    cap = cv2.VideoCapture(0)

    # Set manual exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    # Get the resolution of the webcam feed
    if not cap.isOpened():
        print("Error: Unable to access the webcam")
        return

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define a horizontal line for zone detection
    LINE_START = sv.Point(0, int(0.60 * video_height))
    LINE_END = sv.Point(video_width, int(0.60 * video_height))

    # Create instances of trackers and annotators
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=3,
    )
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK
    )
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=4, text_thickness=4, text_scale=2
    )

    print("Starting webcam feed... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam")
            break

        # Adjust brightness and contrast
        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=5)  # Adjust for overexposure

        # Perform detection
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections)

        # Generate labels
        labels = [
            f"#{tracker_id} {CUSTOM_LABELS.get(model.model.names[class_id], model.model.names[class_id])} {confidence:0.2f}"
            for confidence, class_id, tracker_id in zip(
                detections.confidence, detections.class_id, detections.tracker_id
            )
        ]

        # Annotate the frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        line_zone.trigger(detections)

        annotated_frame = line_zone_annotator.annotate(
            annotated_frame, line_counter=line_zone
        )

        # Display the frame
        cv2.imshow("Webcam Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_webcam()
