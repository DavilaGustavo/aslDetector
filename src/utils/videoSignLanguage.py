import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import eel
import base64
import time
import os
from utils.state_manager import start_execution, is_running

@eel.expose
def videoASL(file_path, saveVideo = 1):
    """
    Function to detect ASL signs in a video file.
    Takes file path as input and processes the video with real-time detection.
    """
    cap = None
    out = None

    try:
        # Starts the execution state
        start_execution()
        
        # Get the directory where the current script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up one directory level if we're in src/
        parent_dir = os.path.dirname(current_dir)

        # Construct the path to the model file
        model_path = os.path.join(parent_dir, 'model', 'model.keras')

        # Load the model with the constructed path
        model = load_model(model_path)

        # Load the video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{file_path}'")
            return

        if saveVideo:
            output_path = os.path.join(parent_dir, 'testTrain', 'outputs')
            os.makedirs(output_path, exist_ok=True)
            
            # Obter propriedades do vídeo original
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Criar arquivo de saída
            output_file = os.path.join(output_path, 'output_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1, max_num_hands=3)

        # ASL alphabet
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
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

            if saveVideo and out is not None:
                out.write(frame)

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
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        eel.clearVideoElement()()