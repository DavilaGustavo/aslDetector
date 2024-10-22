import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('inputs/model.keras')

# Load the Label Encoder from the CSV file
label_mapping = pd.read_csv('inputs/label_encoder.csv')
original_labels = label_mapping['original'].values
encoded_labels = label_mapping['encoded'].values

# Create a new LabelEncoder
label_encoder = LabelEncoder()
label_encoder.classes_ = original_labels  # Set classes directly

# Load the video "aslAlphabet.mp4"
cap = cv2.VideoCapture('inputs/aslAlphabet.mp4')

# Check the size and FPS of the original video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 24

# Create the VideoWriter object to save the video
output_file = 'outputs/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while cap.isOpened():  # Check if the video is open
    ret, frame = cap.read()

    if not ret:  # Check if there are more frames in the video
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect hand coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize the coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Check if we have the correct number of dots (42)
            if len(data_aux) == 42:
                # Transform the input into a numpy array
                data_input = np.asarray(data_aux).reshape(1, -1)

                # Make the prediction
                prediction = model.predict(data_input)
                predicted_index = np.argmax(prediction, axis=1)[0]
                
                # Decode the prediction using the Label Encoder
                predicted_character = label_encoder.inverse_transform([predicted_index])[0]
                predicted_character = alphabet[int(predicted_character)]

                # Draw the letter
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 14,
                            cv2.LINE_AA)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                            cv2.LINE_AA)

    # Write the processed frame to the output file
    out.write(frame)

    cv2.imshow('frame', frame)

    # Add the option to exit with the 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
