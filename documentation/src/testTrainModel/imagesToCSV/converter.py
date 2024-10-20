import os
from PIL import Image
import numpy as np

# Mapeamento das letras para rótulos
label_mapping = {chr(i): i - 65 for i in range(65, 91) if chr(i) not in ['J', 'Z']}  # A=0, B=1, C=2, ..., I=8

# Função para converter e salvar as imagens em dois arquivos CSV
def convert_images_to_csv(data_folder, output_train_csv, output_test_csv, percent):
    # Dicionário para armazenar os dados das imagens por letra
    all_data = {label: [] for label in label_mapping.values()}
    
    # Contador para rótulos 9
    count_label_9 = 0

    # Percorrendo as subpastas
    for label_folder in os.listdir(data_folder):
        label_path = os.path.join(data_folder, label_folder)
        
        # Verificando se é uma pasta
        if os.path.isdir(label_path):
            # Obtendo o rótulo numérico correspondente
            label = label_mapping.get(label_folder)
            if label is not None:
                # Percorrendo as imagens dentro da pasta da letra
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    
                    # Verificando se o arquivo é uma imagem
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            # Abrindo a imagem
                            image = Image.open(image_path)
                            
                            # Convertendo para escala de cinza
                            gray_image = image.convert('L')
                            
                            # Redimensionando a imagem para 64x64
                            resized_image = gray_image.resize((64, 64))
                            
                            # Convertendo a imagem em um array numpy
                            image_array = np.array(resized_image)
                            
                            # Flattening the image array to 1D
                            flattened_array = image_array.flatten()
                            
                            # Adicionando o rótulo e os pixels ao dicionário
                            all_data[label].append(np.insert(flattened_array, 0, label))

                            # Contar quantos rótulos '9' existem
                            if label == 9:
                                count_label_9 += 1

                        except Exception as e:
                            print(f"Erro ao processar a imagem {image_path}: {e}")

    # Exibir quantos rótulos 9 foram encontrados
    print(f"Labels with a '9' (J): {count_label_9}")

    # Listas para armazenar os dados de treino e teste
    train_data = []
    test_data = []

    # Dividindo os dados de cada letra
    for label, data in all_data.items():
        num_test_samples = int(len(data) * 0.1)  # 10% para teste
        num_train_samples = int(len(data[num_test_samples:]) * percent)  # Percentual de treino
        num_test_samples = int(len(data[:num_test_samples]) * percent)  # Percentual de teste

        # Adicionando os dados de treino e teste selecionados
        test_data.extend(data[:num_test_samples])  
        train_data.extend(data[num_test_samples:num_test_samples + num_train_samples])  

    # Embaralhar os dados
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)

    # Convertendo as listas para arrays NumPy
    train_data = np.array(train_data, dtype=object)
    test_data = np.array(test_data, dtype=object)

    # Salvando os arrays em arquivos CSV
    np.savetxt(output_train_csv, train_data, delimiter=',', fmt='%s', header=','.join(['label'] + [f'pixel{i+1}' for i in range(4096)]), comments='')
    np.savetxt(output_test_csv, test_data, delimiter=',', fmt='%s', header=','.join(['label'] + [f'pixel{i+1}' for i in range(4096)]), comments='')

# Exemplo de uso
percent = 0.25
data_folder = '../data/MezclaDatasets'  # Caminho da pasta com as letras
output_train_csv = '../imagesToCSV/asl_alphabet_train.csv'  # Nome do arquivo CSV de saída para treino
output_test_csv = '../imagesToCSV/asl_alphabet_test.csv'  # Nome do arquivo CSV de saída para teste

convert_images_to_csv(data_folder, output_train_csv, output_test_csv, percent)  # Usando 25% dos dados
