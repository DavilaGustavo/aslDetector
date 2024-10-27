import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports for training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Activation

# Load the data
data_df = pd.read_csv('../inputs/data.csv')

# Separate features and labels
data = data_df.iloc[:, :-1].values  # All except the last as data
labels = data_df['label'].values    # The last column as labels

# Encode labels (categories) as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

num_classes = len(np.unique(labels_encoded))

# Split the data into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded, random_state=1337)

# Get the unique values and their respective counts
unique, counts = np.unique(y_train, return_counts=True)

# Create a list of alternating colors
colors = [(116/255, 116/255, 255/255, 1),       # Light purple
          (116/255, 116/255, 255/255, 0.78)]    # Purple
bar_colors = [colors[i % 2] for i in range(len(unique))]    # To better visualization

# Plot the bar chart with the classes
plt.bar(unique, counts, color=bar_colors, tick_label=unique)
plt.title("Number of sign language classes")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("../outputs/sign_language_classes_chart.png")
plt.show()

num_features = 32
epochs = 60
batch_size = 32

# Create the neural network model
model = Sequential()

model.add(Dense(2*num_features, kernel_initializer="he_normal", input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(num_features, kernel_initializer="he_normal"))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(num_features, kernel_initializer="he_normal"))
model.add(Activation('relu'))

model.add(Dense(num_classes, kernel_initializer="he_normal"))
model.add(Activation("softmax"))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

model.summary()

kerasModel = '../inputs/model.keras'
checkpointer = ModelCheckpoint(kerasModel, monitor='val_loss', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

# Train the model
history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[lr_reducer, checkpointer])

# Evaluation
epoch = [i for i in range(epochs)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16,9)

ax[0].plot(epoch , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epoch , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epoch , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epoch , val_loss , 'r-o' , label = 'Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")

plt.savefig("../outputs/training_validation_graph.png")
plt.show()

# Save the Label Encoder
label_mapping = pd.DataFrame({
    'original': label_encoder.classes_,
    'encoded': np.arange(len(label_encoder.classes_))
})

label_mapping.to_csv('../inputs/label_encoder.csv', index=False)