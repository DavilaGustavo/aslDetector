import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import eel
import base64
import os
from utils.state_manager import start_execution, is_running

@eel.expose
def imageASL(file_path):
    """
    Function to detect ASL signs in an image file.
    Takes file path as input and processes the image with ASL detection.
    """
    try:
        # Starts the execution state
        start_execution()
        
        # Get the directory where the current script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Construct the path to the model file
        model_path = os.path.join(parent_dir, 'model', 'model.keras')
        
        # Load the model
        model = load_model(model_path)
        
        # Decode base64 image
        if file_path.startswith('data:image'):
            # Remove the data URL prefix and get only the base64 string
            base64_string = file_path.split(',')[1]
            # Decode base64 string to bytes
            img_data = base64.b64decode(base64_string)
            # Convert bytes to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            frame = cv2.imread(file_path)
            
        if frame is None:
            raise Exception(f"Error: Could not open image file")
            
        # Resize image if too large
        max_dimension = 1280
        height, width = frame.shape[:2]
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1, max_num_hands=10)
        
        # ASL alphabet
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                   'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        
        # Process the image
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract hand landmarks
                data_aux = []
                x_ = []
                y_ = []
                
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                    
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))
                    
                # Make prediction if we have the correct number of landmarks
                if len(data_aux) == 42:
                    data_input = np.asarray(data_aux).reshape(1, -1)
                    prediction = model.predict(data_input)
                    predicted_index = np.argmax(prediction, axis=1)[0]
                    predicted_character = alphabet[predicted_index]
                    
                    # Draw prediction on image
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 14,
                              cv2.LINE_AA)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                              cv2.LINE_AA)
        
        # Convert processed image to base64 for web display
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame to JavaScript
        eel.updateFrame(frame_base64)()
        
    except Exception as e:
        print(f"Error in imageASL: {str(e)}")
    finally:
        if 'hands' in locals():
            hands.close()