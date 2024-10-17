import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports for training
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Load the data ################
dataTrain = pd.read_csv('data/sign_mnist_train/sign_mnist_train.csv')
dataTest = pd.read_csv('data/sign_mnist_test/sign_mnist_test.csv')

#dataTrain.info()
#dataTest.info()

y_train = dataTrain['label']
y_test = dataTest['label']

#print(y_train)

plt.figure(figsize=(15,7))

y_train.value_counts().sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title("Number of sign language classes")
plt.show()

x_train = dataTrain.drop(['label'], axis=1)
x_test = dataTest.drop(['label'], axis=1)

##################################

# Preprocessing the data ###########
# from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# reshape
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)

print(y_train)

#######################################

# Visualization #######################
plt.figure(figsize=(9,7))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.xlabel((y_train[i]))

plt.show()

num_classes = 25 # letters minus J and Z
y_train = to_categorical(y_train, num_classes)

###########################################

# modeling
# Model construction
num_features = 32
width, height = 28, 28
batch_size = 16
epochs = 20

model = Sequential()

# Feature extraction
model.add(Conv2D(num_features, (3, 3), padding='same', kernel_initializer="he_normal",
                 input_shape=(width, height, 1)))
model.add(Activation('elu'))    # testar relu
model.add(BatchNormalization())
model.add(Conv2D(num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Classifier
model.add(Flatten())
model.add(Dense(2*num_features, kernel_initializer="he_normal"))
model.add(Activation('elu'))    # testar relu
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(2*num_features, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(num_classes, kernel_initializer="he_normal"))
model.add(Activation("softmax"))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

model.summary()

# Split into training and validation sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Showing the shapes of our train, validate, and test images
print(Xtrain.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)
print(x_test.shape)
print(y_test.shape)

model_weights_file = "signLanguageModel.keras"  # Model file
model_json_file = "signLanguageModel.json"  # JSON file to save the architecture
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(model_weights_file, monitor='val_loss', verbose=1, save_best_only=True)

# Save the model
model_json = model.to_json()
with open(model_json_file, "w") as json_file:
    json_file.write(model_json)

# Train the model
history = model.fit(np.array(Xtrain), np.array(Ytrain),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(np.array(Xtest), np.array(Ytest)),
        shuffle=True,
        callbacks=[lr_reducer, early_stopper, checkpointer])

# Evaluation

epoch = [i for i in range(20)]
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
plt.show()



















# # Plot a graph showing the total number of images corresponding to each emotion
# plt.figure(figsize=(12,6))
# plt.hist(data['emotion'], bins=6)
# plt.title("Images x Emotion")
# plt.show()
# # Classes: ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Process the data
# pixels = data['pixels'].tolist()
# width, height = 48, 48

# faces = []
# for pixel_sequence in pixels:
#     face = [int(pixel) for pixel in pixel_sequence.split(' ')]
#     face = np.asarray(face).reshape(width, height) 
#     faces.append(face)

# faces = np.asarray(faces) 
# faces = np.expand_dims(faces, -1)

# def normalize(x):
#     x = x.astype('float32')
#     x = x / 255.0
#     return x

# faces = normalize(faces)
# emotions = pd.get_dummies(data['emotion']).to_numpy()

# print("Total number of images in the dataset: " + str(len(faces)))

# # Split into training and validation sets
# x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=41)

# print("Number of images in the training set:", len(x_train))
# print("Number of images in the test set:", len(x_test))
# print("Number of images in the validation set:", len(y_val))

# # Save data for testing
# np.save('mod_xtest', x_test)
# np.save('mod_ytest', y_test)

# # Model construction
# num_features = 32
# num_classes = 7
# width, height = 48, 48
# batch_size = 16
# epochs = 100

# model = Sequential()

# model.add(Conv2D(num_features, (3, 3), padding='same', kernel_initializer="he_normal",
#                  input_shape=(width, height, 1)))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Conv2D(num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Flatten())
# model.add(Dense(2*num_features, kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(2*num_features, kernel_initializer="he_normal"))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(num_classes, kernel_initializer="he_normal"))
# model.add(Activation("softmax"))

# # Compile the model
# model.compile(loss=categorical_crossentropy,
#               optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
#               metrics=['accuracy'])

# model_weights_file = "model_expressions.keras"  # Model file
# model_json_file = "model_expressions.json"  # JSON file to save the architecture
# lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
# early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
# checkpointer = ModelCheckpoint(model_weights_file, monitor='val_loss', verbose=1, save_best_only=True)

# # Save the model
# model_json = model.to_json()
# with open(model_json_file, "w") as json_file:
#     json_file.write(model_json)

# # Train the model
# history = model.fit(np.array(x_train), np.array(y_train),
#         batch_size=batch_size,
#         epochs=epochs,
#         verbose=1,
#         validation_data=(np.array(x_val), np.array(y_val)),
#         shuffle=True,
#         callbacks=[lr_reducer, early_stopper, checkpointer])
