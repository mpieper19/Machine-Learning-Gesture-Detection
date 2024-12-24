import pickle
import cv2
import mediapipe as mp
import numpy as np

model = pickle.load(open('./gesture_model.pkl', 'rb'))

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3
)

labels_dict = {0: 'closed', 1: 'open', 2: 'peace', 3: '3'}

offset = 20

while True:

    success, frame = cam.read()
    frame = cv2.flip(frame, 1)
    height = frame.shape[0]
    width = frame.shape[1]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    norm_data = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                norm_x = landmark.x - min(x_)
                norm_y = landmark.y - min(y_)
                norm_data.append(norm_x)
                norm_data.append(norm_y)

            x_min = int(min(x_) * width) - offset
            x_max = int(max(x_) * width) + offset
            y_min = int(min(y_) * height) - offset
            y_max = int(max(y_) * height) + offset

            if len(norm_data) == 42:
                prediction = model.predict([np.asarray(norm_data)])
                predicted_gesture = labels_dict[int(prediction[0])]

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    1
                )
                cv2.putText(
                    frame,
                    predicted_gesture,
                    (x_min, y_min - offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow('Gesture Recognition', frame)
    key = cv2.waitKey(1)

    # Quit on 'q' key press
    if key == ord('q'):
        break
