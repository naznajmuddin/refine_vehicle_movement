# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# Set up home directory
PROJECT_PATH = os.getcwd()

# Load YOLOv8 model for CPU
model = YOLO("yolov8n.pt")  # Initialize the model
model.to("cpu")  # Explicitly set the model to use the CPU

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

# Video source path
SOURCE_VIDEO_PATH = os.path.join(PROJECT_PATH, "test7.mp4")


# Single frame prediction and annotation
def annotate_single_frame():
    # Create frame generator and instance of annotators
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK
    )

    # Acquire first video frame
    iterator = iter(generator)
    frame = next(iterator)
    frame = cv2.resize(frame, (640, 360))  # Reduce resolution for better speed

    # Model prediction on single frame
    results = model(frame, conf=0.4, verbose=False)[0]  # Lower confidence threshold
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

    # Format custom labels
    labels = [
        f"{CUSTOM_LABELS.get(CLASS_NAMES_DICT[class_id], CLASS_NAMES_DICT[class_id])} {confidence:0.5f}"
        for confidence, class_id in zip(detections.confidence, detections.class_id)
    ]

    # Annotate and display frame
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    # Plot image (using matplotlib)
    sv.plot_image(annotated_frame, (16, 16))


# Full video processing and annotation
def test_video_real_time():

    # Get video resolution
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_width = video_info.width
    video_height = video_info.height

    # LINE
    LINE_START = sv.Point(0, int(0.60 * video_height))
    LINE_END = sv.Point(video_width, int(0.60 * video_height))

    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # Create instances of tracking and annotators
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.5,  # Higher threshold for fewer tracks
        lost_track_buffer=20,  # Reduce buffer size
        minimum_matching_threshold=0.7,  # Lower matching threshold
        frame_rate=15,  # Lower frame rate for smoother processing
        minimum_consecutive_frames=2,  # Reduce consecutive frame requirement
    )
    byte_tracker.reset()

    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(
        text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK
    )
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=4, text_thickness=4, text_scale=2
    )

    frame_count = 0  # Frame skipping counter

    # Process video frame-by-frame in real-time
    for frame in generator:
        frame_count += 1
        if frame_count % 2 != 0:  # Skip every other frame
            continue

        frame = cv2.resize(frame, (1280, 720))  # Resize to lower resolution

        results = model(frame, conf=0.4, verbose=False)[0]  # Lower confidence threshold
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {CUSTOM_LABELS.get(model.model.names[class_id], model.model.names[class_id])} {confidence:0.2f}"
            for confidence, class_id, tracker_id in zip(
                detections.confidence, detections.class_id, detections.tracker_id
            )
        ]

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

        # Display the frame in real-time
        cv2.imshow("YOLO Real-Time Detection", annotated_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    annotate_single_frame()
    test_video_real_time()
