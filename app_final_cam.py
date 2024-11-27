import os
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Define selected class names for tracking
SELECTED_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
model = YOLO("yolov8s.pt")
CLASS_NAMES_DICT = model.model.names
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

# Global variables for webcam and shared frame
cap = cv2.VideoCapture(0)  # Initialize webcam
stop_thread = False  # To signal the thread to stop
annotated_frame = None  # Shared variable for the annotated frame
frame_lock = threading.Lock()  # Lock to synchronize frame access


# GUI function
def process_gui():
    def update_frame():
        global annotated_frame, stop_thread

        with frame_lock:
            if annotated_frame is not None:
                # Convert the annotated frame to RGB format for tkinter
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the label with the annotated frame
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)

        if not stop_thread:
            video_label.after(10, update_frame)  # Schedule next frame update

    # Create the main tkinter window
    root = tk.Tk()
    root.title("Real-Time Object Detection")
    root.geometry("1000x600")
    root.configure(bg="#fff")

    # Create a main frame
    main_frame = tk.Frame(root, bg="#fff")
    main_frame.pack(fill="both", expand=True)

    # Left frame for video playback
    video_frame = tk.Frame(main_frame, bg="#000", width=700, height=600)
    video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    video_label = tk.Label(video_frame, bg="#000")
    video_label.pack(fill="both", expand=True)

    # Right frame for vehicle counter
    counter_frame = tk.Frame(main_frame, bg="#d4cfcf", width=300, height=600)
    counter_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    title_label = tk.Label(
        counter_frame,
        text="Jumlah Kendaraan",
        font=("Arial", 14, "bold"),
        bg="#d4cfcf",
    )
    title_label.pack(pady=(0, 10))

    count_labels = []
    for vehicle in vehicle_counts:
        label = tk.Label(
            counter_frame,
            text=f"{vehicle} Masuk: {vehicle_counts[vehicle]['in']}",
            font=("Arial", 14, "bold"),
            bg="#d4cfcf",
            anchor="w",
        )
        label.pack(anchor="w", pady=5)
        count_labels.append(label)

    total_label = tk.Label(
        counter_frame,
        text=f"Total di Lokasi: {total_count}",
        font=("Arial", 14, "bold"),
        bg="#d4cfcf",
    )
    total_label.pack(pady=10)

    def update_labels():
        global total_count
        for idx, (vehicle, counts) in enumerate(vehicle_counts.items()):
            count_labels[idx].config(
                text=f"{vehicle} Masuk: {counts['in']}"
            )

        total_label.config(text=f"Total di Lokasi: {total_count}")
        root.after(1000, update_labels)

    update_labels()

    # Configure grid weights for responsiveness
    main_frame.columnconfigure(0, weight=3)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)

    # Start updating frames
    update_frame()

    # Handle application close
    def on_closing():
        global stop_thread
        stop_thread = True
        cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


# Real-time video processing function
def process_webcam_feed():
    global cap, vehicle_counts, total_count, stop_thread, annotated_frame

    print("Processing webcam feed...")
    byte_tracker = sv.ByteTrack()
    byte_tracker.reset()

    # Get video dimensions from OpenCV
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the line for counting
    line_zone = sv.LineZone(
        start=sv.Point(0, int(0.65 * video_height)),
        end=sv.Point(video_width, int(0.65 * video_height)),
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1, text_scale=0.5, text_color=sv.Color.WHITE
    )
    trace_annotator = sv.TraceAnnotator(
        trace_length=50, thickness=2
    )  # Add trace annotator
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2, text_scale=0.5, text_thickness=1
    )

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections)

        # Annotate the frame
        annotated_frame_local = frame.copy()
        annotated_frame_local = trace_annotator.annotate(  # Annotate traces
            scene=annotated_frame_local, detections=detections
        )
        annotated_frame_local = box_annotator.annotate(
            scene=annotated_frame_local, detections=detections
        )
        annotated_frame_local = label_annotator.annotate(
            scene=annotated_frame_local,
            detections=detections,
            labels=[
                f"{CLASS_NAMES_DICT[class_id]}: {CUSTOM_LABELS.get(class_id, 'N/A')}"
                for class_id in detections.class_id
            ],
        )
        annotated_frame_local = line_zone_annotator.annotate(
            annotated_frame_local, line_counter=line_zone
        )

        # Update counts
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

        # Share the annotated frame with the GUI
        with frame_lock:
            annotated_frame = annotated_frame_local


if __name__ == "__main__":
    gui_thread = threading.Thread(target=process_gui)
    video_thread = threading.Thread(target=process_webcam_feed)

    gui_thread.start()
    video_thread.start()

    gui_thread.join()
    video_thread.join()
