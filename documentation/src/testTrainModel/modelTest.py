import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image  # Importar para manipulação de imagens
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Carregar o modelo salvo
json_file = open('signLanguageModel.json', 'r')
model_json = json_file.read()
json_file.close()

# Carregar a arquitetura do modelo
model = model_from_json(model_json)

# Carregar os pesos salvos
model.load_weights('signLanguageModel.keras')

# Compilar o modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Preprocessar os dados de teste
dataTest = pd.read_csv('data/asl_alphabet_test.csv')
y_test = dataTest['label']

x_test = dataTest.drop(['label'], axis=1)

# Normalizar os dados
x_test = x_test / 255.0

# Remodelar os dados para (64, 64, 1)
x_test = x_test.values.reshape(-1, 64, 64, 1)

# Mapeamento de números para letras do alfabeto (sem o J e o Z)
num_to_char = {i: chr(65+i) for i in range(25)}  # 65 é 'A' na tabela ASCII

# Fazer predição no conjunto de testes
y_pred = np.argmax(model.predict(x_test), axis=1)

y_test_alpha = [num_to_char[val] for val in y_test]
y_pred_alpha = [num_to_char[val] for val in y_pred]

# Criar a matriz de confusão
conf_matrix = confusion_matrix(y_test_alpha, y_pred_alpha)

# Exibir a matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=num_to_char.values(), yticklabels=num_to_char.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Salvar a matriz de confusão em um arquivo
plt.savefig("outputs/confusion_matrix.png")  # Salva a matriz de confusão como PNG
plt.show()

accuracy = accuracy_score(y_test_alpha, y_pred_alpha)
report = classification_report(y_test_alpha, y_pred_alpha)

print(report)
print(accuracy)

# Exibir algumas imagens de teste e suas predições
random_indices = np.random.choice(range(len(x_test)), size=10, replace=False)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.xlabel(f"Actual: {y_test_alpha[idx]}\nPredicted: {y_pred_alpha[idx]}")

plt.tight_layout()
plt.show()

# Parte extra: identificar uma imagem "teste.png"
def preprocess_image(image_path):
    """Carrega, redimensiona e normaliza a imagem para o formato do modelo."""
    img = Image.open(image_path).convert('L')  # Abrir a imagem e converter para escala de cinza
    img = img.resize((64, 64))  # Redimensionar para 28x28
    img = np.array(img)  # Converter para um array NumPy
    img = img / 255.0  # Normalizar os valores dos pixels (0-1)
    img = img.reshape(1, 64, 64, 1)  # Ajustar para o formato (1, 28, 28, 1) para o modelo
    return img

# Adicionar gráfico 3x3 com as imagens teste11.png a teste33.png
image_paths = [
    'inputs/test.png', 'inputs/test2.png', 'inputs/test3.png',
    'inputs/teste21.png', 'inputs/teste22.png', 'inputs/teste23.png',
    'inputs/teste31.png', 'inputs/teste32.png', 'inputs/teste33.png'
]

plt.figure(figsize=(10, 10))

# Fazer predições para as 9 imagens e exibi-las
for i, image_path in enumerate(image_paths):
    processed_image = preprocess_image(image_path)  # Pré-processar cada imagem
    
    # Fazer a predição
    prediction = np.argmax(model.predict(processed_image), axis=1)
    prediction_alpha = num_to_char[prediction[0]]
    
    # Exibir a imagem e a predição no gráfico
    plt.subplot(3, 3, i + 1)
    plt.imshow(processed_image.reshape(64, 64), cmap='gray')
    plt.title(f"Predicted: {prediction_alpha}")
    plt.axis('off')  # Ocultar os eixos

# Ajustar layout e exibir a imagem final
plt.tight_layout()
plt.savefig("outputs/predicted_images_3x3.png")  # Salvar o gráfico final como uma imagem PNG
plt.show()
