import os
import cv2

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Define path for data storage
data_path = os.path.join(parent_dir, 'data')

# Print paths for debugging
# print(f"\nDirectories used:")
# print(f"Current directory: {current_dir}")
# print(f"Parent directory: {parent_dir}")
# print(f"Data will be saved to: {data_path}")

# If data folder doesnt exist, creates it (/data)
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"Created directory: {data_path}")

# Set the amount of classes/images to get from the webcam
classes = 25        # A-Z without Z, removing J after extraction
amountFrames = 800  # 800 images each letter

cap = cv2.VideoCapture(0) # '2' if '0' dont work

for i in range(classes):
    # Create class directory path
    class_path = os.path.join(data_path, str(i))
    
    # If specific data folder doesnt exist, creates it (/data/0, 1, 2, 3...)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
        print(f"Created directory for class {i}: {class_path}")

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
        
        # Save image with robust path
        image_path = os.path.join(class_path, f'{j}.jpg')
        cv2.imwrite(image_path, frame)

cap.release()
cv2.destroyAllWindows()