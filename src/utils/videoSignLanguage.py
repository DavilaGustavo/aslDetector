import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import eel
import base64
import time
from utils.state_manager import start_execution, is_running

@eel.expose
def videoASL(file_path):
    """
    Function to detect ASL signs in a video file.
    Takes file path as input and processes the video with real-time detection.
    """
    cap = None
    
    try:
        # Inicia o estado de execução
        start_execution()
        
        # Load the trained model
        model = load_model('src/model/model.keras')

        # Load the video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{file_path}'")
            return

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # ASL alphabet without J
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

        while is_running() and cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Resize frame if too large
            max_dimension = 1280
            height, width = frame.shape[:2]
            if height > max_dimension or width > max_dimension:
                scale = max_dimension / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    data_aux = []
                    x_ = []
                    y_ = []

                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                    if len(data_aux) == 42:
                        data_input = np.asarray(data_aux).reshape(1, -1)
                        prediction = model.predict(data_input)
                        predicted_index = np.argmax(prediction, axis=1)[0]
                        predicted_character = alphabet[predicted_index]

                        x1 = int(min(x_) * W) - 10
                        y1 = int(min(y_) * H) - 10
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 14,
                                cv2.LINE_AA)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                                cv2.LINE_AA)

            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame to JavaScript
            eel.updateFrame(frame_base64)()

            # Control frame rate
            time.sleep(1/30)  # Limit to 30 FPS

    except Exception as e:
        print(f"Error in videoASL: {str(e)}")
        
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # Limpa a imagem no frontend
        eel.clearVideoElement()()