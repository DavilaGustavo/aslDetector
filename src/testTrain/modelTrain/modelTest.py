import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import load_model
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

# Load the test data
test_df = pd.read_csv(os.path.join(input_path, 'dataTest.csv'))

# Separate features and labels
x_test = test_df.iloc[:, :-1].values  # All except the last as data
y_test = test_df['label'].values    # The last column as labels

# Ensure labels are numeric
try:
    y_test = y_test.astype(int)
except:
    print("Labels are not numeric, attempting to convert from strings...")
    # If labels are letters, convert them to indices
    label_to_index = {label: idx for idx, label in enumerate(alphabet)}
    y_test = np.array([label_to_index[label] for label in y_test])

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

# Select a random sample from test data using numpy
random_idx = np.random.randint(0, len(x_test))
random_sample = x_test[random_idx]
true_label = alphabet[y_test[random_idx]]
predicted_label = alphabet[y_pred[random_idx]]

# Create a new figure for the hand visualization with smaller size
plt.figure(figsize=(8, 8))
plt.grid(True)

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]

# Extract x and y coordinates (42 points -> 21 pairs)
x_coords = random_sample[::2]
y_coords = random_sample[1::2]

# Invert coordinates
x_coords = 1 - x_coords  # Invert X axis
y_coords = 1 - y_coords  # Invert Y axis

# Plot the connections
for start_idx, end_idx in HAND_CONNECTIONS:
    plt.plot([x_coords[start_idx], x_coords[end_idx]], 
             [y_coords[start_idx], y_coords[end_idx]], 
             'r--', linewidth=1)

# Plot the landmarks
plt.scatter(x_coords, y_coords, c='red', s=72)

# Add landmark numbers for reference (with smaller font)
for i in range(21):
    plt.annotate(str(i), (x_coords[i], y_coords[i]), 
                xytext=(3, 3), textcoords='offset points', fontsize=8)

# Customize the plot
plt.title(f'Hand Landmarks\nTrue Label: {true_label} | Predicted: {predicted_label}')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# Make the plot square and set limits
plt.axis('equal')
margin = 0.05
plt.xlim(min(x_coords) - margin, max(x_coords) + margin)
plt.ylim(min(y_coords) - margin, max(y_coords) + margin)

# Save the visualization
output_image_file = os.path.join(output_path, 'random_test_sample.png')
plt.savefig(output_image_file, bbox_inches='tight')
plt.show()