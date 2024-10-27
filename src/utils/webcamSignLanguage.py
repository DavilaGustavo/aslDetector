import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import eel
import base64
from utils.state_manager import start_execution, is_running

@eel.expose
def webcamASL(resolution=(1280, 720), max_hands=4, camera_index=0):
    """
    Function to detect ASL signs using webcam
    Parameters:
        resolution (tuple): Width and height of camera resolution (default: 720p)
        max_hands (int): Maximum number of hands to detect (default: 4)
        camera_index (int): Camera device index (default: 0)
    """
    cap = None
    
    try:
        # Inicia o estado de execução
        start_execution()
        
        # Load the trained model
        model = load_model('src/model/model.keras')

        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        if not cap.isOpened():
            print("Error: Could not open webcam")
            eel.handleWebcamError()()
            return

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, 
                            min_detection_confidence=0.3,
                            max_num_hands=max_hands)

        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

        while is_running():  # Usa o controle de estado
            ret, frame = cap.read()

            if not ret:
                break

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
                
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []
                    
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)
                    
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

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

            # Convert frame to JPEG and then to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame to JavaScript
            eel.updateFrame(frame_base64)()

    except Exception as e:
        print(f"Error in webcamASL: {str(e)}")
        eel.handleWebcamError()()
    
    finally:
        # Cleanup sempre será executado
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # Limpa a imagem no frontend
        eel.clearVideoElement()()