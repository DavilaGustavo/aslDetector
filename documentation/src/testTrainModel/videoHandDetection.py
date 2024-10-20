import cv2
import time
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Variáveis para ajustes
totalHands = 3      # Aumente ou diminua a quantidade de mãos a detectar
square_scale = 1.4  # Aumente ou diminua para alterar o tamanho do quadrado
bones = False       # Mostrar ou não as ligações que identificam a mão
detectionColour = (25, 200, 70)

# Inicializar MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=totalHands)
mp_drawing = mp.solutions.drawing_utils  # Para desenhar as landmarks

# Carregar o modelo treinado
model = load_model('signLanguageModel.keras')

# Definir o alfabeto das classes previstas
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Abrir o arquivo de vídeo
video_file = "handTest3.mp4"
cap = cv2.VideoCapture(video_file)

# Definir o arquivo de saída
output_file = 'handTestResultVideo.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Função para preprocessar a imagem de entrada para MediaPipe e o modelo
def preprocess_image(img, hand_landmarks):
    h, w, _ = img.shape
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
    hand_img_resized = cv2.resize(hand_img, (64, 64))

    # Converter para escala de cinza e normalizar
    hand_img_gray = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2GRAY)
    hand_img_normalized = hand_img_gray / 255.0

    # Reshape para o formato esperado pelo modelo
    hand_img_reshaped = hand_img_normalized.reshape(-1, 64, 64, 1)

    return hand_img_reshaped, x_min_square, y_min_square  # Retornar a imagem processada e as coordenadas do quadrado

# Loop para capturar cada quadro do vídeo
while cap.isOpened():
    connected, frame = cap.read()
    
    if not connected:
        break

    # Converter o quadro para RGB (necessário para MediaPipe)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar mãos no quadro
    results = hands.process(img_rgb)

    # Verificar se mãos foram detectadas
    if results.multi_hand_landmarks:
        # Processar cada mão detectada
        for hand_landmarks in results.multi_hand_landmarks:
            # Preprocessar a área da mão detectada para o formato que o modelo espera
            hand_img, x_min_square, y_min_square = preprocess_image(frame, hand_landmarks)

            # Fazer a predição com o modelo treinado
            prediction = model.predict(hand_img)
            predicted_class = np.argmax(prediction)  # Retorna a classe mais provável

            # Converter a predição numérica para a letra correspondente
            predicted_letter = alphabet[predicted_class]

            # Desenhar landmarks e conexões na imagem
            if bones:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Exibir a predição na imagem
            cv2.putText(frame, f"Predicted: {predicted_letter}", 
                        (x_min_square + 5, y_min_square + 20),  # Posição ajustada para dentro do quadrado
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, detectionColour, 2)

    # Escrever o quadro processado no arquivo de saída
    output_video.write(frame)

    # Mostrar o quadro processado (opcional)
    cv2.imshow('Frame', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
output_video.release()
cv2.destroyAllWindows()
