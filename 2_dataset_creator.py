import csv
import os
import mediapipe as mp
import cv2

# Path to the directory containing captured frames
captured_frames = "./captured_frames"

# Initialize MediaPipe Hands module for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create and write headers for CSV file
with open('gesture_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = [
        'ID', '0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y',
        '4_x', '4_y', '5_x', '5_y',  '6_x', '6_y', '7_x', '7_y', '8_x',
        '8_y', '9_x', '9_y', '10_x', '10_y', '11_x', '11_y',  '12_x',
        '12_y', '13_x', '13_y', '14_x', '14_y', '15_x', '15_y', '16_x',
        '16_y', '17_x', '17_y', '18_x', '18_y', '19_x', '19_y', '20_x',
        '20_y'
    ]

    writer.writerow(field)

# Process each image in the captured frames directory
for folder in os.listdir(captured_frames):
    for image_path in os.listdir(os.path.join(captured_frames, folder)):
        # Read and preprocess image
        image = cv2.imread(os.path.join(captured_frames, folder, image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                row = [folder]
                x_ = []
                y_ = []

                # Extract landmark coordinates
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize the coordinates
                for landmark in hand_landmarks.landmark:
                    norm_x = landmark.x - min(x_)
                    norm_y = landmark.y - min(y_)
                    row.append(norm_x)
                    row.append(norm_y)

                # Append processed data to gseture_data.csv
                with open("gesture_data.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
