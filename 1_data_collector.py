import cv2
import os
import time

cam = cv2.VideoCapture(0)

current_folder = None
output_dir = "captured_frames"

is_capturing = False
capture_start_time = None

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if cam.isOpened():
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(width, height)

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    offset = 50
    img_size = 600

    cv2.putText(
        frame,
        "Press a number (0-9) to create a folder.",
        (5, 25),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (0, 255, 255),
        1
    )
    cv2.putText(
        frame,
        "Press 'q' to quit. ",
        (5, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    if current_folder:
        cv2.putText(
            frame,
            f"Current Folder: {current_folder}",
            (5, 75),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        cv2.putText(
            frame,
            "Prepare your hand gesture, and press 'c' to capture for 30 seconds.",
            (5, 100),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    else:
        cv2.putText(
            frame,
            "No folder selected!",
            (5, 75),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    if is_capturing:
        capture_end_time = time.time()
        elapsed_time = capture_end_time - capture_start_time
        cv2.putText(
            frame,
            f"Capturing... {30 - int(elapsed_time)} seconds left",
            (5, 435),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        frame_count = len(os.listdir(current_folder))
        frame_name = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(current_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        cv2.putText(
            frame,
            f"Captued {frame_name}",
            (5, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        if elapsed_time >= 30:
            is_capturing = False
            print("Finished capturing frames for 30 seconds.")

    cv2.imshow('Data Collector', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif ord('0') <= key <= ord('9'):
        folder_name = chr(key)
        current_folder = os.path.join(output_dir, folder_name)

        if not os.path.exists(current_folder):
            os.mkdir(current_folder)
            print(f"Create folder: {current_folder}")
        else:
            print(f"Folder already exists: {current_folder}")
    elif key == ord('c'):
        if current_folder:
            is_capturing = True
            capture_start_time = time.time()
            print("Started capturing frames for 30 seconds...")
        else:
            print("No folder selected. Please select a folder by pressing a number key (0-9).")
