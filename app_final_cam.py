import os
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Constants
PROJECT_PATH = os.getcwd()
SOURCE_VIDEO_PATH = os.path.join(PROJECT_PATH, "test6.mp4")
TARGET_VIDEO_PATH = os.path.join(PROJECT_PATH, "result.mp4")

# Initialize YOLO model
model = YOLO("yolov8m.pt")
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

CUSTOM_LABELS = {
    2: "mobil",
    3: "motor",
    5: "bus",
    7: "truck",
}

vehicle_counts = {vehicle: {"in": 0, "out": 0} for vehicle in CUSTOM_LABELS.values()}
total_count = 0

# Global Variables
cap = None  # Webcam capture object
stop_thread = False
annotated_frame = None
frame_lock = threading.Lock()


# Helper function to process and annotate frames


def process_frame(
    frame, line_zone, byte_tracker, box_annotator, label_annotator, line_zone_annotator
):
    global vehicle_counts, total_count

    # Run YOLO model inference
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter detections for selected class IDs
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)

    # Update line zone counters
    line_zone.trigger(detections)
    for class_id, in_count in line_zone.out_count_per_class.items():
        vehicle_counts[CUSTOM_LABELS.get(class_id, f"Class {class_id}")][
            "in"
        ] = in_count

    for class_id, out_count in line_zone.in_count_per_class.items():
        vehicle_counts[CUSTOM_LABELS.get(class_id, f"Class {class_id}")][
            "out"
        ] = out_count

    # Calculate total vehicles inside
    total_count = sum(counts["in"] for counts in vehicle_counts.values()) - sum(
        counts["out"] for counts in vehicle_counts.values()
    )

    # Log the trace information for each class
    for class_id, counts in vehicle_counts.items():
        print(f"Class: {class_id} | In: {counts['in']} | Out: {counts['out']}")

    # Annotate the frame
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=[
            f"{CLASS_NAMES_DICT[class_id]}: {CUSTOM_LABELS.get(class_id, 'N/A')}"
            for class_id in detections.class_id
        ],
    )
    annotated_frame = line_zone_annotator.annotate(
        frame=annotated_frame, line_counter=line_zone
    )

    # Add trace annotations to the frame
    for i, (vehicle, counts) in enumerate(vehicle_counts.items()):
        cv2.putText(
            annotated_frame,
            f"{vehicle.capitalize()} - In: {counts['in']} | Out: {counts['out']}",
            (10, 30 + i * 20),  # Position for each line
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # Font scale
            (255, 255, 255),  # Text color (white)
            2,  # Thickness
            cv2.LINE_AA,
        )

    return annotated_frame


# GUI Function
# GUI Function
def create_gui():
    def start_webcam():
        global cap, stop_thread
        stop_thread = False
        cap = cv2.VideoCapture(0)

        byte_tracker = sv.ByteTrack()
        byte_tracker.reset()

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        line_zone = sv.LineZone(
            start=sv.Point(0, int(0.65 * video_height)),
            end=sv.Point(video_width, int(0.65 * video_height)),
        )
        box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
        label_annotator = sv.LabelAnnotator(
            text_thickness=1, text_scale=0.5, text_color=sv.Color.WHITE
        )
        line_zone_annotator = sv.LineZoneAnnotator(
            thickness=2, text_scale=0.5, color=sv.Color.WHITE
        )

        def update_webcam_frame():
            global cap, annotated_frame, stop_thread

            if stop_thread or cap is None:
                return

            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    annotated_frame = process_frame(
                        frame,
                        line_zone,
                        byte_tracker,
                        box_annotator,
                        label_annotator,
                        line_zone_annotator,
                    )

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)

            video_label.after(10, update_webcam_frame)

        update_webcam_frame()

    def stop_webcam():
        global cap, stop_thread
        stop_thread = True
        if cap is not None:
            cap.release()

    def update_labels():
        for idx, (vehicle, counts) in enumerate(vehicle_counts.items()):
            count_labels[idx].config(text=f"{vehicle.capitalize()}: {counts['in']}")
        total_label.config(text=f"Total di Lokasi: {total_count}")
        root.after(1000, update_labels)

    # Main GUI Window
    root = tk.Tk()
    root.title("Intelligent Surveillance System di Fakultas Teknik")
    root.geometry("1200x700")
    root.configure(bg="#f4f4f4")

    # Header
    header_frame = tk.Frame(root, bg="#d0c6c5", height=60)
    header_frame.pack(fill="x", side="top")

    header_label = tk.Label(
        header_frame,
        text="Intelligent Surveillance System di Fakultas Teknik",
        font=("Arial", 24, "bold"),
        bg="#d0c6c5",
        fg="#000",
    )
    header_label.pack(pady=10)

    # Main layout
    main_frame = tk.Frame(root, bg="#f4f4f4", padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)

    # Video display
    video_frame = tk.Frame(main_frame, bg="#000", width=800, height=600)
    video_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 20), pady=0)

    video_label = tk.Label(video_frame, bg="#000")
    video_label.pack(fill="both", expand=True)

    # Counter and controls
    counter_frame = tk.Frame(main_frame, bg="#d0c6c5", width=350, height=600)
    counter_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

    # Vehicle count display
    count_title = tk.Label(
        counter_frame,
        text="Jumlah Kendaraan Masuk",
        font=("Arial", 16, "bold"),
        bg="#d0c6c5",
        fg="#000",
    )
    count_title.pack(anchor="center", pady=10)

    count_labels = []
    for vehicle in vehicle_counts:
        count_text = f"{vehicle.capitalize()}: {vehicle_counts[vehicle]['in']}"
        count_label = tk.Label(
            counter_frame,
            text=count_text,
            font=("Arial", 14),
            bg="#d0c6c5",
            fg="#333",
            anchor="w",
        )
        count_label.pack(anchor="center", pady=5)
        count_labels.append(count_label)

    # Add total count label
    total_label = tk.Label(
        counter_frame,
        text=f"Total di Lokasi: {total_count}",
        font=("Arial", 14, "bold"),
        bg="#d0c6c5",
        fg="#000",
    )
    total_label.pack(anchor="center", pady=10)

    # Control buttons
    button_frame = tk.Frame(counter_frame, bg="#d0c6c5")
    button_frame.pack(fill="x", pady=20)

    webcam_button = tk.Button(
        button_frame,
        text="Start Webcam",
        font=("Arial", 12),
        command=start_webcam,
        bg="#d0c6c5",
        fg="#000",
    )
    webcam_button.pack(anchor="center")

    # Configure weights
    main_frame.columnconfigure(0, weight=3)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)

    update_labels()
    root.mainloop()


if __name__ == "__main__":
    create_gui()
