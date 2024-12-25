import cv2
import os
import time

# Initialize webcam for capturing video feed
cam = cv2.VideoCapture(0)

# Variables for managing output folder and capturing states
current_folder = None
output_dir = "captured_frames"  # Directory to save captured frames
is_capturing = False  # Flag to track capture status
capture_start_time = None  # To record the start time of capturing

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Check if the webcam is opened successfully
if cam.isOpened():
    # Retrieve and display the webcam resolution
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)  # Print webcam resolution

while True:
    # Capture frames from the webcam
    success, frame = cam.read()
    if not success:  # If frame not captured, exit loop
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Display instructions on the video feed
    cv2.putText(
        frame, "Press a number (0-9) to create a folder.", (5, 25),
        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1
    )
    cv2.putText(
        frame, "Press 'q' to quit. ", (5, 50),
        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1
    )

    # Display selected folder name if available
    if current_folder:
        cv2.putText(
            frame, f"Current Folder: {current_folder}", (5, 75),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1
        )
        cv2.putText(
            frame,
            "Prepare your hand gesture, and press 'c' to capture for 30 seconds.",
            (5, 100),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1
        )
    else:
        # Show warning if no folder is selected
        cv2.putText(
            frame, "No folder selected!", (5, 75),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1
        )

    # Handle frame capturing if enabled
    if is_capturing:
        # Calculate elapsed time
        capture_end_time = time.time()
        elapsed_time = capture_end_time - capture_start_time

        # Show countdown timer
        cv2.putText(
            frame, f"Capturing... {30 - int(elapsed_time)} seconds left",
            (5, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

        # Save captured frame in the selected folder
        frame_count = len(os.listdir(current_folder))  # Count frames in folder
        frame_name = f"frame_{frame_count}.jpg"  # Generate frame name
        frame_path = os.path.join(current_folder, frame_name)  # File path
        cv2.imwrite(frame_path, frame)  # Save the frame

        # Display saved frame status
        cv2.putText(
            frame, f"Captured {frame_name}", (5, 455),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

        # Stop capturing after 30 seconds
        if elapsed_time >= 30:
            is_capturing = False
            print("Finished capturing frames for 30 seconds.")

    # Show the video feed
    cv2.imshow('Data Collector', frame)

    # Capture user key presses
    key = cv2.waitKey(1)

    # Quit the program if 'q' is pressed
    if key == ord('q'):
        break
    # Create folder when a number key (0-9) is pressed
    elif ord('0') <= key <= ord('9'):
        folder_name = chr(key)  # Get folder name as the pressed key
        current_folder = os.path.join(output_dir, folder_name)

        # Create folder if it doesn't exist
        if not os.path.exists(current_folder):
            os.mkdir(current_folder)
            print(f"Created folder: {current_folder}")
        else:
            print(f"Folder already exists: {current_folder}")
    # Start capturing frames when 'c' is pressed
    elif key == ord('c'):
        if current_folder:  # Ensure a folder is selected
            is_capturing = True
            capture_start_time = time.time()  # Record start time
            print("Started capturing frames for 30 seconds...")
        else:
            print("No folder selected. Please select a folder by pressing a number key (0-9).")

# Release webcam and close OpenCV windows
cam.release()
cv2.destroyAllWindows()
