import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import seaborn as sns

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_parent_dir = os.path.dirname(parent_dir)

# Define paths
input_path = os.path.join(parent_dir, 'inputs')
output_path = os.path.join(parent_dir, 'outputs')
model_path = os.path.join(model_parent_dir, 'model')

# Create directories if they don't exist
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Load the trained model
model = load_model(os.path.join(model_path, 'model.keras'))

# Defines the alphabet
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Load the data for evaluation
data_df = pd.read_csv(os.path.join(input_path, 'data.csv'))

# Separate features and labels
data = data_df.iloc[:, :-1].values  # All except the last as data
labels = data_df['label'].values    # The last column as labels

# Ensure labels are numeric
try:
    labels = labels.astype(int)
except:
    print("Labels are not numeric, attempting to convert from strings...")
    # If labels are letters, convert them to indices
    label_to_index = {label: idx for idx, label in enumerate(alphabet)}
    labels = np.array([label_to_index[label] for label in labels])

# Split the data into training/testing sets with a random seed
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, 
    test_size=0.2, 
    shuffle=True, 
    stratify=labels, 
    random_state=1337
)

# Load a test image
image_file = os.path.join(input_path, 'handTest.png')
image = cv2.imread(image_file)

# Initialize mediapipe to detect hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand detection with parameters
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=10, 
    min_detection_confidence=0.02
)

# Gets the height and weight
H, W, _ = image.shape

# Converts the image to rgb (mediapipe only works with rgb)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image
results = hands.process(image_rgb)

# Check if there's hands being detected
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        # Draw landmarks in the hands
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Get the coordinates of the hand
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

        # Check if there's 42 dots on the hand
        if len(data_aux) == 42:
            # Reshape the input to a array numpy
            data_input = np.asarray(data_aux).reshape(1, -1)

            # Do the prediction
            prediction = model.predict(data_input)
            predicted_index = np.argmax(prediction, axis=1)[0]
            
            # Get the predicted character directly from alphabet
            predicted_character = alphabet[predicted_index]

            # Draw the letter
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            cv2.putText(image, predicted_character, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 14, cv2.LINE_AA)
            cv2.putText(image, predicted_character, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

# Show the image
cv2.imshow('Hand Sign Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Do predictions on the test set
y_pred = np.argmax(model.predict(x_test), axis=1)

# Creates a confusion matrix using numeric labels
conf_matrix = confusion_matrix(y_test, y_pred)

# Define alphabet without J for visualization
alphabet_no_j = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Show the matrix with letter labels
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=alphabet_no_j, yticklabels=alphabet_no_j)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Save the matrix
plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
plt.show()

# Convert numeric labels to letters for the classification report
y_test_letters = [alphabet[idx] for idx in y_test]
y_pred_letters = [alphabet[idx] for idx in y_pred]

# Calculate and display metrics
accuracy = accuracy_score(y_test_letters, y_pred_letters)
report = classification_report(y_test_letters, y_pred_letters)

print(report)
print(f"Accuracy: {accuracy:.4f}")

# Save the initial image after the prediction
output_image_file = os.path.join(output_path, 'output_image.png')
cv2.imwrite(output_image_file, image)

# Print paths for debugging
# print(f"\nDirectories used:")
# print(f"Current directory: {current_dir}")
# print(f"Parent directory: {parent_dir}")
# print(f"Model parent directory: {model_parent_dir}")
# print(f"Input path: {input_path}")
# print(f"Output path: {output_path}")
# print(f"Model path: {model_path}")