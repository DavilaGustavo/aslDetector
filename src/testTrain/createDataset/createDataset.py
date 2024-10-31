import os
import pandas as pd
import mediapipe as mp
import cv2

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Define paths
data_path = os.path.join(parent_dir, 'data')
input_path = os.path.join(parent_dir, 'inputs')

# Create input directory if it doesn't exist
os.makedirs(input_path, exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []
for dir_ in os.listdir(data_path):
    dir_path = os.path.join(data_path, dir_)
    
    # Check if it's a directory
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Check if there's 42 dots to guarantee consistency (amount of dots for each hand with mediapipe)
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)

# Creates a dataframe
df = pd.DataFrame(data)
df['label'] = labels

# Save the dataframe
df.to_csv(os.path.join(input_path, 'dataTemporario.csv'), index=False)

# Print paths for debugging
# print(f"\nDirectories used:")
# print(f"Current directory: {current_dir}")
# print(f"Parent directory: {parent_dir}")
# print(f"Data path: {data_path}")
# print(f"Input path: {input_path}")
# print(f"Data will be saved to: {os.path.join(input_path, 'data.csv')}")