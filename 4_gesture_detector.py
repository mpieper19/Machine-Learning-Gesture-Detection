import pickle
import cv2
import mediapipe as mp
import numpy as np 

# Load the pre-trained gesture recognition model
model = pickle.load(open('./gesture_model.pkl', 'rb'))

# Initialize webcam for live video feed
cam = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module for hand detection and tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure MediaPipe Hands parameters
hands = mp_hands.Hands(
    static_image_mode=False,  # Detect hands in a video stream
    max_num_hands=1,  # Detect a maximum of one hand
    min_detection_confidence=0.3  # Confidence threshold for detection
)

# Define labels corresponding to gestures detected by the model
## Edit this dictionary with yoursaved gestures!!
labels_dict = {0: 'closed', 1: 'open', 2: 'peace', 3: '3'}

# Padding around detected hand regions
offset = 20

while True:
    # Read frames from webcam
    success, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip horizontally for a mirror effect
    height, width = frame.shape[:2]  # Get frame dimensions

    # Convert BGR image to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands and landmarks
    results = hands.process(frame_rgb)

    # Prepare data for gesture prediction
    norm_data = []
    x_ = []  # To store x-coordinates
    y_ = []  # To store y-coordinates

    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the hand
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract x and y coordinates of landmarks
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize the coordinates relative to the hand's position
            for landmark in hand_landmarks.landmark:
                norm_x = landmark.x - min(x_)  # Normalize x
                norm_y = landmark.y - min(y_)  # Normalize y
                norm_data.append(norm_x)
                norm_data.append(norm_y)

            # Calculate bounding box for detected hand
            x_min = int(min(x_) * width) - offset
            x_max = int(max(x_) * width) + offset
            y_min = int(min(y_) * height) - offset
            y_max = int(max(y_) * height) + offset

            # Predict gesture if valid data is collected
            if len(norm_data) == 42:  # Ensure 42 features for prediction
                prediction = model.predict([np.asarray(norm_data)])
                predicted_gesture = labels_dict[int(prediction[0])]  # Map prediction to label

                # Draw bounding box and label on the frame
                cv2.rectangle(
                    frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1
                )
                cv2.putText(
                    frame, predicted_gesture,
                    (x_min, y_min - offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

    # Display the processed frame
    cv2.imshow('Gesture Recognition', frame)

    # Exit loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release webcam and close OpenCV windows
cam.release()
cv2.destroyAllWindows()
