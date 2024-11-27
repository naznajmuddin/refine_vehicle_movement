import os
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

video_ready_event = threading.Event()
cap = None

PROJECT_PATH = os.getcwd()
SOURCE_VIDEO_PATH = os.path.join(PROJECT_PATH, "test6.mp4")
TARGET_VIDEO_PATH = os.path.join(PROJECT_PATH, "result.mp4")

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

CUSTOM_ANNOTATE_LABELS = {
    "car": "mobil",
    "motorcycle": "motor",
    "bus": "bus",
    "truck": "truck",
}

vehicle_counts = {vehicle: {"in": 0, "out": 0} for vehicle in CUSTOM_LABELS.values()}
total_count = 0


def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    points = [
        x1 + radius,
        y1,
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def process_gui():
    def play_video(replay=False):
        global cap
        if replay or cap is None:
            cap = cv2.VideoCapture(TARGET_VIDEO_PATH)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = int(1000 / fps)

        def update_frame():
            global cap
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
                video_label.after(frame_delay, update_frame)
            else:
                cap.release()
                replay_video()

        update_frame()

    def replay_video():
        play_video(replay=True)

    root = tk.Tk()
    root.title("Intelligent Surveillance System di Fakultas Teknik")
    root.geometry("1200x700")
    root.configure(bg="#f4f4f4")

    # Header
    header_frame = tk.Frame(root, bg="#eaeaea", height=60)
    header_frame.pack(fill="x", side="top")

    header_label = tk.Label(
        header_frame,
        text="Intelligent Surveillance System di Fakultas Teknik",
        font=("Arial", 24, "bold"),
        bg="#eaeaea",
        fg="#000",
    )
    header_label.pack(pady=10)

    # Main layout frame
    main_frame = tk.Frame(root, bg="#f4f4f4", padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)

    # Video frame
    video_frame = tk.Frame(main_frame, bg="#000", width=800, height=600)
    video_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 20), pady=0)

    video_label = tk.Label(video_frame, bg="#000")
    video_label.pack(fill="both", expand=True)

    # Counter frame
    counter_frame = tk.Frame(main_frame, bg="#fff", width=350, height=600)
    counter_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

    # IP Display
    ip_frame = tk.Frame(counter_frame, bg="#d0c6c5", padx=10, pady=10)
    ip_frame.pack(fill="x", pady=(0, 10))

    ip_label = tk.Label(
        ip_frame,
        text="IP Cam: 192.168.233.100",
        font=("Arial", 14, "bold"),
        bg="#d0c6c5",
        fg="#333",
    )
    ip_label.pack(anchor="center")

    # Vehicle Count Display
    count_frame = tk.Frame(counter_frame, bg="#d0c6c5", padx=20, pady=20)
    count_frame.pack(fill="both", expand=True)

    count_title = tk.Label(
        count_frame,
        text="Jumlah Kendaraan Masuk",
        font=("Arial", 16, "bold"),
        bg="#d0c6c5",
        fg="#000",
    )
    count_title.pack(anchor="center")

    count_labels = []
    for vehicle in vehicle_counts:
        count_text = f"{vehicle.capitalize()} : {vehicle_counts[vehicle]['in']}"
        count_label = tk.Label(
            count_frame,
            text=count_text,
            font=("Arial", 14),
            bg="#d0c6c5",
            fg="#333",
            anchor="w",
        )
        count_label.pack(anchor="center", pady=5)
        count_labels.append(count_label)

    def update_labels():
        for idx, (vehicle, counts) in enumerate(vehicle_counts.items()):
            count_labels[idx].config(text=f"{vehicle.capitalize()}: {counts['in']}")
        root.after(1000, update_labels)

    update_labels()

    # Configure column and row weights for resizing
    main_frame.columnconfigure(0, weight=3)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)

    def auto_play_video():
        video_ready_event.wait()
        play_video()

    threading.Thread(target=auto_play_video, daemon=True).start()

    root.mainloop()


def process_whole_video():
    global vehicle_counts, total_count
    print("Processing the video...")

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_width = video_info.width
    video_height = video_info.height

    LINE_START = sv.Point(0, int(0.65 * video_height))
    LINE_END = sv.Point(video_width, int(0.65 * video_height))

    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
    line_annotator = sv.LineZoneAnnotator(thickness=2, color=sv.Color.WHITE)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1, text_scale=0.5, text_color=sv.Color.WHITE
    )

    byte_tracker = sv.ByteTrack()
    byte_tracker.reset()
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        global total_count
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections)

        line_zone.trigger(detections)
        for class_id, out_count in line_zone.in_count_per_class.items():
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["out"] = out_count

        for class_id, in_count in line_zone.out_count_per_class.items():
            class_label = CUSTOM_LABELS.get(int(class_id), f"Class {class_id}")
            vehicle_counts[class_label]["in"] = in_count

        labels = [
            f"{CUSTOM_ANNOTATE_LABELS.get(CLASS_NAMES_DICT[class_id], CLASS_NAMES_DICT[class_id])}"
            for confidence, class_id in zip(detections.confidence, detections.class_id)
        ]

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        line_annotator.annotate(frame=annotated_frame, line_counter=line_zone)

        for idx, (vehicle, counts) in enumerate(vehicle_counts.items()):
            text = f"{vehicle}: Masuk: {counts['in']} | Keluar: {counts['out']}"
            position = (10, 30 + idx * 20)
            cv2.putText(
                annotated_frame,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return annotated_frame

    sv.process_video(
        source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=callback
    )

    video_ready_event.set()


if __name__ == "__main__":
    gui_thread = threading.Thread(target=process_gui)
    video_thread = threading.Thread(target=process_whole_video)

    gui_thread.start()
    video_thread.start()

    gui_thread.join()
    video_thread.join()
