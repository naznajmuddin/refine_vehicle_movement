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

    print("Processing the video...")
    # Get video resolution
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_width = video_info.width
    video_height = video_info.height

    # LINE
    LINE_START = sv.Point(0, int(0.65 * video_height))
    LINE_END = sv.Point(video_width, int(0.65 * video_height))

    TARGET_VIDEO_PATH = os.path.join(PROJECT_PATH, "result.mp4")

    # Create instances of track and annotation classes
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=3,
    )
    byte_tracker.reset()

    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(
        text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK
    )
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=1, text_thickness=1, text_scale=1, show_text=False
    )

    # Counter for each vehicle type
    vehicle_counts = {
        vehicle: {"in": 0, "out": 0} for vehicle in CUSTOM_LABELS.values()
    }

    # Initialize a total count variable for vehicles within the premise

    total_count = 0

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        nonlocal total_count  # To keep track of the total count across frames
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections)

        # Trigger the line zone to detect crossings
        line_zone.trigger(detections)

        # Update vehicle counts and adjust the total count
        for (
            class_id,
            out_count,
        ) in line_zone.in_count_per_class.items():  # Switch in to out
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["out"] = out_count

        for (
            class_id,
            in_count,
        ) in line_zone.out_count_per_class.items():  # Switch out to in
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["in"] = in_count

        # Calculate the net total count (vehicles inside the premise)
        current_total_in = sum(
            line_zone.out_count_per_class.values()
        )  # Use out_count as in
        current_total_out = sum(
            line_zone.in_count_per_class.values()
        )  # Use in_count as out
        total_count = current_total_in - current_total_out

        # Debug: Print the updated vehicle counts in the terminal
        print("Jumlah Kendaraan:")
        print(vehicle_counts)
        print(f"Total kendaraan di lokasi: {total_count}")

        # Create labels for detections
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

        # Display counts on the frame (top-left information)
        count_text = "\n".join(
            [
                f"{vehicle}: Masuk={counts['in']} Keluar={counts['out']}"
                for vehicle, counts in vehicle_counts.items()
            ]
        )
        total_text = f"Total kendaraan di lokasi: {total_count}"

        # Define the overlay dimensions
        overlay = annotated_frame.copy()
        alpha = 0.5  # Transparency level
        overlay_x_start = 30
        overlay_y_start = 30
        overlay_x_end = 400
        overlay_y_end = 200  # Adjust height based on the text content

        # Draw a semi-transparent rectangle for the top-left text
        cv2.rectangle(
            overlay,
            (overlay_x_start, overlay_y_start),
            (overlay_x_end, overlay_y_end),
            (0, 0, 0),  # Black background
            -1,  # Filled
        )
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # Draw the top-left text
        y_start = overlay_y_start + 30
        for line in count_text.split("\n") + [total_text]:
            cv2.putText(
                annotated_frame,
                line,
                (overlay_x_start + 10, y_start),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Font scale
                (255, 255, 255),  # White text color
                2,  # Thickness
            )
            y_start += 30

        # Create a semi-transparent overlay for the "Masuk" and "Keluar" text
        overlay = annotated_frame.copy()
        alpha = 0.5  # Transparency level (0: fully transparent, 1: opaque)

        # Define the text for "Masuk" and "Keluar"
        masuk_text = f"Masuk: {sum(line_zone.out_count_per_class.values())}"  # Use out_count as in
        keluar_text = f"Keluar: {sum(line_zone.in_count_per_class.values())}"  # Use in_count as out

        # Position for the "Masuk" and "Keluar" text
        masuk_position = (50, annotated_frame.shape[0] - 100)  # Bottom-left
        keluar_position = (50, annotated_frame.shape[0] - 50)  # Just above "Masuk"

        # Draw filled rectangles as background for the text
        cv2.rectangle(
            overlay,
            (30, annotated_frame.shape[0] - 140),
            (400, annotated_frame.shape[0] - 30),
            (0, 0, 0),
            -1,
        )

        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # Draw the "Masuk" and "Keluar" text on the transparent overlay
        cv2.putText(
            annotated_frame,
            masuk_text,
            masuk_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated_frame,
            keluar_text,
            keluar_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return annotated_frame

    # Process video
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=callback
    )


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
