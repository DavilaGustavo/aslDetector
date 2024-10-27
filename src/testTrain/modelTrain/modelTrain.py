import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports for training
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization

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

# Define the alphabet for ASL
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Load the data
data_df = pd.read_csv(os.path.join(input_path, 'data.csv'))

# Separate features and labels
data = data_df.iloc[:, :-1].values  # All except the last as data
labels = data_df['label'].values    # The last column as labels

# Split the data into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, 
    test_size=0.2, 
    shuffle=True, 
    stratify=labels, 
    random_state=1337
)

# Get the unique values and their respective counts
unique, counts = np.unique(y_train, return_counts=True)

# Create a list of alternating colors
colors = [(116/255, 116/255, 255/255, 1),       # Light purple
          (116/255, 116/255, 255/255, 0.78)]    # Purple
bar_colors = [colors[i % 2] for i in range(len(unique))]    # To better visualization

# Plot the bar chart with the classes
plt.figure(figsize=(12, 6))
plt.bar(range(len(counts)), counts, color=bar_colors)
plt.title("Distribution of ASL Letters in Training Set")
plt.xlabel("ASL Letters")
plt.ylabel("Count")
plt.xticks(range(len(unique)), unique, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'asl_letters_distribution.png'))
plt.show()

# Model configuration
num_features = 64  # Increased number of features
epochs = 60
batch_size = 32

# Create an improved neural network model
model = Sequential([
    # Input layer with batch normalization
    Dense(2*num_features, kernel_initializer="he_normal", input_shape=(x_train.shape[1],)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),
    
    # Hidden layers with residual-like connections
    Dense(num_features, kernel_initializer="he_normal"),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),
    
    Dense(num_features//2, kernel_initializer="he_normal"),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.1),
    
    # Output layer
    Dense(len(alphabet), kernel_initializer="he_normal"),
    Activation("softmax")
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
    metrics=['accuracy']
)

model.summary()

# Setup callbacks
keras_model_path = os.path.join(model_path, 'model.keras')
checkpointer = ModelCheckpoint(
    keras_model_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.9,
    patience=3,
    verbose=1
)

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    shuffle=True,
    callbacks=[lr_reducer, checkpointer]
)

# Evaluation plots
epoch = range(epochs)
fig, ax = plt.subplots(1, 2, figsize=(16, 9))

# Training and validation metrics
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Accuracy plot
ax[0].plot(epoch, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epoch, val_acc, 'ro-', label='Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

# Loss plot
ax[1].plot(epoch, train_loss, 'g-o', label='Training Loss')
ax[1].plot(epoch, val_loss, 'r-o', label='Testing Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")

plt.savefig(os.path.join(output_path, 'training_validation_graph.png'))
plt.show()

# Print some information about the paths
# print(f"\nDirectories used:")
# print(f"Current directory: {current_dir}")
# print(f"Parent directory: {parent_dir}")
# print(f"Model parent directory: {model_parent_dir}")
# print(f"Model path: {model_path}")