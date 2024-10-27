# Code mainly focused on creating a dataset by getting frames out of the webcam
# can be used to any image based data creation but made to be used with ASL

import os
import cv2

dataFolder = '../data'

# If data folder doesnt exist, creates it (/data)
if not os.path.exists(dataFolder):
    os.makedirs(dataFolder)

# Set the amount of classes/images to get from the webcam
classes = 25        # A-Z without Z, removing J after extraction
amountFrames = 800  # 800 images each letter

cap = cv2.VideoCapture(0) # '2' if '0' dont work

for i in range(classes):
    # If specific data folder doesnt exist, creates it (/data/0, 1, 2, 3...)
    if not os.path.exists(os.path.join(dataFolder, str(i))):
        os.makedirs(os.path.join(dataFolder, str(i)))

    print('Current class: {}'.format(i))

    # Retains before collection the data
    while True:
        ret, frame = cap.read()
        text = f'Q to start. Class: {format(i)}'
        cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 14, cv2.LINE_AA)
        cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Pass by amountFrames to get images from
    for j in range(amountFrames):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(5) # Change to make it slower/faster
        cv2.imwrite(os.path.join(dataFolder, str(i), '{}.jpg'.format(j)), frame)

cap.release()
cv2.destroyAllWindows()