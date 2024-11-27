# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO
import supervision as sv
import numpy as np
import platform
import subprocess
import cv2

# Set up home directory
PROJECT_PATH = os.getcwd()

# Load YOLOv8 model
model = YOLO("yolov8m.pt")
CLASS_NAMES_DICT = model.model.names

# Define selected class names for tracking
SELECTED_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# Mapping for custom class labels
CUSTOM_LABELS = {
    2: "mobil",  # Class 2 corresponds to "mobil"
    3: "motor",  # Class 3 corresponds to "motor"
    5: "bus",  # Class 5 corresponds to "bus"
    7: "truck",  # Class 7 corresponds to "truck"
}

CUSTOM_ANNOTATE_LABELS = {
    "car": "mobil",
    "motorcycle": "motor",
    "bus": "bus",
    "truck": "truck",
}


# Download the example video
from supervision.assets import download_assets, VideoAssets

# SOURCE_VIDEO_PATH = download_assets(VideoAssets.VEHICLES)
SOURCE_VIDEO_PATH = os.path.join(PROJECT_PATH, "test6.mp4")


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

    # Model prediction on single frame
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

    # Format custom labels
    labels = [
        f"{CUSTOM_ANNOTATE_LABELS.get(CLASS_NAMES_DICT[class_id], CLASS_NAMES_DICT[class_id])} {confidence:0.5f}"
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
def process_whole_video():
    global vehicle_counts, total_count

    print("Processing the video...")

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_width = video_info.width
    video_height = video_info.height

    # Define a line for counting "Masuk" and "Keluar"
    LINE_START = sv.Point(0, int(0.65 * video_height))
    LINE_END = sv.Point(video_width, int(0.65 * video_height))

    byte_tracker = sv.ByteTrack()
    byte_tracker.reset()

    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(
        text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK
    )
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=1, text_thickness=1, text_scale=1, show_text=False
    )

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        global total_count
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections)

        # Update "Masuk" and "Keluar" counts
        line_zone.trigger(detections)
        for class_id, out_count in line_zone.in_count_per_class.items():
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["out"] = out_count

        for class_id, in_count in line_zone.out_count_per_class.items():
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["in"] = in_count

        # Calculate total vehicles inside
        current_total_in = sum(line_zone.out_count_per_class.values())
        current_total_out = sum(line_zone.in_count_per_class.values())
        total_count = current_total_in - current_total_out

        # Format labels
        labels = [
            f"{CUSTOM_ANNOTATE_LABELS.get(CLASS_NAMES_DICT[class_id], CLASS_NAMES_DICT[class_id])}"
            for class_id in detections.class_id
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
        annotated_frame = line_zone_annotator.annotate(
            annotated_frame, line_counter=line_zone
        )

        # Overlay vehicle counts
        overlay = annotated_frame.copy()
        alpha = 0.6  # Transparency level
        cv2.rectangle(
            overlay,
            (30, annotated_frame.shape[0] - 140),  # Bottom-left corner
            (400, annotated_frame.shape[0] - 30),  # Adjust size of the overlay
            (0, 0, 0),  # Black background
            -1,  # Filled
        )
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # Add "Masuk" and "Keluar" counts
        masuk_text = f"Masuk: {sum(line_zone.out_count_per_class.values())}"
        keluar_text = f"Keluar: {sum(line_zone.in_count_per_class.values())}"

        cv2.putText(
            annotated_frame,
            masuk_text,
            (50, annotated_frame.shape[0] - 100),  # Position for "Masuk"
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  # White text color
            2,
        )
        cv2.putText(
            annotated_frame,
            keluar_text,
            (50, annotated_frame.shape[0] - 50),  # Position for "Keluar"
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  # White text color
            2,
        )

        # Add total vehicles text
        total_text = f"Total di lokasi: {total_count}"
        cv2.putText(
            annotated_frame,
            total_text,
            (50, annotated_frame.shape[0] - 150),  # Position for total vehicles
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  # White text color
            2,
        )

        return annotated_frame

    sv.process_video(
        source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=callback
    )

    # Signal that video processing is complete
    video_ready_event.set()


if __name__ == "__main__":
    annotate_single_frame()
    process_whole_video()

    TARGET_VIDEO_PATH = os.path.join(PROJECT_PATH, "result.mp4")

    # Open the video with the default video player
    if platform.system() == "Windows":
        os.startfile(TARGET_VIDEO_PATH)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", TARGET_VIDEO_PATH])
    else:  # Linux and others
        subprocess.call(["xdg-open", TARGET_VIDEO_PATH])
