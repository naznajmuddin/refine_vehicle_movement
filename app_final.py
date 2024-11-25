import os
from ultralytics import YOLO
import supervision as sv
import numpy as np
import platform
import subprocess
import cv2
import tkinter as tk
import threading

PROJECT_PATH = os.getcwd()

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
    2: "mobil",
    3: "motor",
    5: "bus",
    7: "truck",
}

CUSTOM_ANNOTATE_LABELS = {
    "car": "mobil",
    "motorcycle": "motor",
    "bus": "bus",
    "truck": "truck",
}

SOURCE_VIDEO_PATH = os.path.join(PROJECT_PATH, "test6.mp4")
TARGET_VIDEO_PATH = os.path.join(PROJECT_PATH, "result.mp4")

# Global variables for GUI display
vehicle_counts = {vehicle: {"in": 0, "out": 0} for vehicle in CUSTOM_LABELS.values()}
total_count = 0


# GUI function
def process_gui():
    # Create a new tkinter window
    root = tk.Tk()
    root.title("Intelligent Surveillance System")
    root.geometry("400x300")
    root.configure(bg="#fff")  # Light gray background

    # Add IP Cam label
    ip_label = tk.Label(
        root,
        text="IP Cam: 192.168.233.100",
        font=("Arial", 12),
        bg="#d4cfcf",  # Light beige color
        width=50,
        anchor="w",
        padx=10,
        pady=5,
    )
    ip_label.pack(pady=(10, 5))

    # Add container for vehicle counts
    container = tk.Frame(root, bg="#d4cfcf", padx=20, pady=20)
    container.pack(pady=(5, 10))

    # Add title for the counts section
    title_label = tk.Label(
        container,
        text="Jumlah Kendaraan Masuk",
        font=("Arial", 14, "bold"),
        bg="#d4cfcf",
    )
    title_label.pack(pady=(0, 10))

    # Add labels for vehicle counts
    count_labels = []
    for vehicle in vehicle_counts:
        label = tk.Label(
            container,
            text=f"{vehicle}: {vehicle_counts[vehicle]['in']}",
            font=("Arial", 12),
            bg="#d4cfcf",
            anchor="w",
        )
        label.pack(anchor="w")
        count_labels.append(label)

    # Function to update the labels
    def update_labels():
        # Update the counts dynamically
        for idx, (vehicle, counts) in enumerate(vehicle_counts.items()):
            count_labels[idx].config(text=f"{vehicle}: {counts['in']}")

        # Schedule the function to run again after 1 second
        root.after(1000, update_labels)

    # Start the update loop
    update_labels()

    # Start the tkinter main loop
    root.mainloop()


# Full video processing and annotation
def process_whole_video():
    global vehicle_counts, total_count

    print("Processing the video...")

    # Get video resolution
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_width = video_info.width
    video_height = video_info.height

    # Define line zone
    LINE_START = sv.Point(0, int(0.65 * video_height))
    LINE_END = sv.Point(video_width, int(0.65 * video_height))

    # Create instances of tracking and annotation classes
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

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        global total_count
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections)

        # Trigger the line zone to detect crossings
        line_zone.trigger(detections)

        # Update vehicle counts and total count
        for class_id, out_count in line_zone.in_count_per_class.items():
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["out"] = out_count

        for class_id, in_count in line_zone.out_count_per_class.items():
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["in"] = in_count

        # Calculate the net total count
        current_total_in = sum(line_zone.out_count_per_class.values())
        current_total_out = sum(line_zone.in_count_per_class.values())
        total_count = current_total_in - current_total_out

        # Terminal output for debugging
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

        # Add bottom-left information (Masuk/Keluar)
        overlay = annotated_frame.copy()
        alpha = 0.5

        masuk_text = f"Masuk: {current_total_in}"  # Use out_count as in
        keluar_text = f"Keluar: {current_total_out}"  # Use in_count as out

        masuk_position = (50, annotated_frame.shape[0] - 100)  # Bottom-left
        keluar_position = (50, annotated_frame.shape[0] - 50)  # Above "Masuk"

        # Semi-transparent rectangle for bottom-left text
        cv2.rectangle(
            overlay,
            (30, annotated_frame.shape[0] - 140),
            (400, annotated_frame.shape[0] - 30),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # Draw "Masuk" and "Keluar" text
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

    # Open the video with the default video player
    if platform.system() == "Windows":
        os.startfile(TARGET_VIDEO_PATH)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", TARGET_VIDEO_PATH])
    else:  # Linux and others
        subprocess.call(["xdg-open", TARGET_VIDEO_PATH])


if __name__ == "__main__":
    # Create threads for the GUI and video processing
    gui_thread = threading.Thread(target=process_gui)
    video_thread = threading.Thread(target=process_whole_video)

    # Start the threads
    gui_thread.start()
    video_thread.start()

    # Wait for both threads to complete
    gui_thread.join()
    video_thread.join()
