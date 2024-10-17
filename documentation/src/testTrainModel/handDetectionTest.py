import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import model_from_json

# Inicializar MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Carregar o modelo salvo
json_file = open('signLanguageModel.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('signLanguageModel.keras')

# Variável para ajustar o tamanho do quadrado
square_scale = 1.5  # Aumente ou diminua para alterar o tamanho do quadrado

# Preprocessamento da imagem de entrada para MediaPipe e o modelo
def preprocess_image(img, hand_landmarks):
    h, w, c = img.shape
    # Pegar as coordenadas da mão (bounding box ao redor da mão)
    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

    # Calcular a altura e largura da bounding box
    box_height = y_max - y_min
    box_width = x_max - x_min

    # Calcular o novo tamanho para o quadrado
    square_size = int(max(box_height, box_width) * square_scale)

    # Calcular as novas coordenadas para o quadrado
    x_center = x_min + box_width // 2
    y_center = y_min + box_height // 2

    x_min_square = int(max(0, x_center - square_size // 2))
    y_min_square = int(max(0, y_center - square_size // 2))
    x_max_square = int(min(w, x_center + square_size // 2))
    y_max_square = int(min(h, y_center + square_size // 2))

    # Desenhar retângulo ao redor da mão (agora um quadrado)
    cv2.rectangle(img, (x_min_square, y_min_square), (x_max_square, y_max_square), (0, 255, 0), 2)

    # Recortar a área da mão
    hand_img = img[y_min_square:y_max_square, x_min_square:x_max_square]

    # Redimensionar para o tamanho esperado pelo modelo (28x28)
    hand_img_resized = cv2.resize(hand_img, (28, 28))

    # Converter para escala de cinza e normalizar
    hand_img_gray = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2GRAY)
    hand_img_normalized = hand_img_gray / 255.0

    # Reshape para o formato esperado pelo modelo
    hand_img_reshaped = hand_img_normalized.reshape(1, 28, 28, 1)

    return hand_img_reshaped  # Retornar apenas a imagem processada

# Carregar a imagem de teste
img = cv2.imread('image2.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detectar a mão na imagem
results = hands.process(img_rgb)

# Verificar se a mão foi detectada
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Preprocessar a área da mão detectada
        hand_img = preprocess_image(img, hand_landmarks)

        # Fazer a predição com o modelo treinado
        prediction = model.predict(hand_img)
        predicted_class = np.argmax(prediction)

        # Converter a predição numérica para a letra correspondente
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        # Converter a predição numérica para a letra correspondente
        if predicted_class < len(alphabet):
            predicted_letter = alphabet[predicted_class]
        else:
            predicted_letter = "Unknown"  # Ou outra mensagem de erro adequada

        print("Prediction:", prediction)
        print("Predicted class index:", predicted_class)

        
        # Exibir a predição na imagem original
        cv2.putText(img, f"Predicted: {predicted_letter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Exibir a imagem final com as predições (sem os processamentos mostrados)
cv2.imshow('Hand Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
