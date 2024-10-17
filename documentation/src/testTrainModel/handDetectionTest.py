import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import model_from_json

# Variáveis para ajustes
totalHands = 2      # Aumente ou diminua a quantidade de mãos a detectar
square_scale = 1.2  # Aumente ou diminua para alterar o tamanho do quadrado
detectionColour = (25,25,230)
imageFile = 'image5.png'

# Inicializar MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=totalHands)
mp_drawing = mp.solutions.drawing_utils

# Carregar o modelo salvo
json_file = open('signLanguageModel.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('signLanguageModel.keras')

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
    cv2.rectangle(img, (x_min_square, y_min_square), (x_max_square, y_max_square), detectionColour, 2)

    # Recortar a área da mão
    hand_img = img[y_min_square:y_max_square, x_min_square:x_max_square]

    # Redimensionar para o tamanho esperado pelo modelo (28x28)
    hand_img_resized = cv2.resize(hand_img, (28, 28))

    # Converter para escala de cinza e normalizar
    hand_img_gray = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2GRAY)
    hand_img_normalized = hand_img_gray / 255.0

    # Reshape para o formato esperado pelo modelo
    hand_img_reshaped = hand_img_normalized.reshape(-1, 28, 28, 1)

    return hand_img_reshaped, x_min_square, y_min_square  # Retornar a imagem processada e as coordenadas do quadrado

# Carregar a imagem de teste
img = cv2.imread(imageFile)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Detectar a mão na imagem
results = hands.process(img_rgb)

# Verificar se a mão foi detectada
if results.multi_hand_landmarks:
    # Passa por cada mão
    for hand_landmarks in results.multi_hand_landmarks:
        # Preprocessar a área da mão detectada para o formato que o modelo espera
        hand_img, x_min_square, y_min_square = preprocess_image(img, hand_landmarks) # 28x28 cinza

        # Fazer a predição com o modelo treinado
        prediction = model.predict(hand_img)
        predicted_class = np.argmax(prediction) # Retorna qual o mais provavel das letras

        # Converter a predição numérica para a letra correspondente
        predicted_letter = alphabet[predicted_class]

        #print("Prediction:", prediction)
        print("Predicted class index:", predicted_class)

        # Exibir a predição acima do quadrado ao redor da mão
        cv2.putText(img, f"Predicted: {predicted_letter}", 
                    (x_min_square, y_min_square - 10),  # Posição ajustada para cima do quadrado
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, detectionColour, 2)

# Exibir a imagem final com as predições (sem os processamentos mostrados)
cv2.imshow('Hand Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
