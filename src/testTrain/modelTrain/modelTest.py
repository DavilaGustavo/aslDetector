import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the trained model
model = load_model('../inputs/model.keras')

# Load the label encoder
label_mapping = pd.read_csv('../inputs/label_encoder.csv')
original_labels = label_mapping['original'].values
encoded_labels = label_mapping['encoded'].values

# Creates a new label encoder (there's probably a better way to do this)
label_encoder = LabelEncoder()
label_encoder.classes_ = original_labels  # Define the classes directly

# Load the data for evaluation
data_df = pd.read_csv('../inputs/data.csv')

# Separate features and labels
data = data_df.iloc[:, :-1].values  # All except the last as data
labels = data_df['label'].values    # The last column as labels

# Encode labels (categories) as integers
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training/testing sets with a random seed
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded, random_state=1337)

# Load a test image
image_file = "../inputs/handTest.png"
image = cv2.imread(image_file)

# Initialize mediapipe to detect hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=10, min_detection_confidence=0.02) # Probably change the min_detection_confidence

# Defines the alphabet (without Z and should not have J)
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

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
            
            # Decode the prediction
            predicted_character = label_encoder.inverse_transform([predicted_index])[0] ## Check a better way to do this
            predicted_character = alphabet[int(predicted_character)]

            # Draw the letter
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            cv2.putText(image, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 14, cv2.LINE_AA)
            cv2.putText(image, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

# Show the image
cv2.imshow('Hand Sign Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

alphabet_adjusted = [char for i, char in enumerate(alphabet) if i != 9]

# Do a prediction on the test set
y_pred = np.argmax(model.predict(x_test), axis=1)

y_test_alpha = [alphabet[val] for val in y_test]
y_pred_alpha = [alphabet[val] for val in y_pred]

# Creates a confusion matrix
conf_matrix = confusion_matrix(y_test_alpha, y_pred_alpha)

# Show the matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=alphabet_adjusted, yticklabels=alphabet_adjusted)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Save the matrix
plt.savefig("../outputs/confusion_matrix.png")
plt.show()

accuracy = accuracy_score(y_test_alpha, y_pred_alpha)
report = classification_report(y_test_alpha, y_pred_alpha)

print(report)
print(accuracy)

# Save the initial image after the prediction
output_image_file = "../outputs/output_image.png"
cv2.imwrite(output_image_file, image)