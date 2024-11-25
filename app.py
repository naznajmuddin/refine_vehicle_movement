import os
import threading
import tkinter as tk
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2


# GUI Class
class CounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Counters")
        self.root.geometry("400x300")

        # Create labels for vehicle counts
        self.labels = {}
        for vehicle in CUSTOM_LABELS.values():
            label = tk.Label(
                self.root,
                text=f"{vehicle.capitalize()}: Masuk=0, Keluar=0",
                font=("Helvetica", 12),
            )
            label.pack(pady=5)
            self.labels[vehicle] = label

        # Create label for total count
        self.total_label = tk.Label(
            self.root, text="Total Kendaraan: 0", font=("Helvetica", 14, "bold")
        )
        self.total_label.pack(pady=20)

    def update_counts(self, vehicle_counts, total_count):
        # Update the GUI labels with the new counts
        for vehicle, counts in vehicle_counts.items():
            self.labels[vehicle].config(
                text=f"{vehicle.capitalize()}: Masuk={counts['in']}, Keluar={counts['out']}"
            )
        self.total_label.config(text=f"Total Kendaraan: {total_count}")


# Video Processing Function
def process_video_with_gui(gui):
    print("Processing the video...")

    # Video resolution and line settings
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_width = video_info.width
    video_height = video_info.height
    LINE_START = sv.Point(0, int(0.65 * video_height))
    LINE_END = sv.Point(video_width, int(0.65 * video_height))
    TARGET_VIDEO_PATH = os.path.join(PROJECT_PATH, "result_with_gui.mp4")

    # Create instances of track and annotation classes
    byte_tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
    byte_tracker.reset()
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
    box_annotator = sv.BoxAnnotator(thickness=4)
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

    vehicle_counts = {
        vehicle: {"in": 0, "out": 0} for vehicle in CUSTOM_LABELS.values()
    }
    total_count = 0

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        nonlocal total_count

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections)
        line_zone.trigger(detections)

        # Update vehicle counts
        for class_id, out_count in line_zone.in_count_per_class.items():
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["out"] = out_count
        for class_id, in_count in line_zone.out_count_per_class.items():
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["in"] = in_count

        # Update total count
        current_total_in = sum(line_zone.out_count_per_class.values())
        current_total_out = sum(line_zone.in_count_per_class.values())
        total_count = current_total_in - current_total_out

        # Update GUI in real-time
        gui.update_counts(vehicle_counts, total_count)

        # Annotate the frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        return annotated_frame

    sv.process_video(
        source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=callback
    )


# Main Program
if __name__ == "__main__":
    # Initialize YOLO Model and Settings
    PROJECT_PATH = os.getcwd()
    model = YOLO("yolov8m.pt")
    CLASS_NAMES_DICT = model.model.names
    SELECTED_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
    SELECTED_CLASS_IDS = [
        {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
        for class_name in SELECTED_CLASS_NAMES
    ]
    CUSTOM_LABELS = {2: "mobil", 3: "motor", 5: "bus", 7: "truck"}
    SOURCE_VIDEO_PATH = os.path.join(PROJECT_PATH, "test6.mp4")

    # Start GUI in a separate thread
    root = tk.Tk()
    gui = CounterGUI(root)
    video_thread = threading.Thread(target=process_video_with_gui, args=(gui,))
    video_thread.start()

    # Run the GUI main loop
    root.mainloop()
    video_thread.join()
