import cv2
import matplotlib.pyplot as plt
import os

def display_image(image_path):
    """
    Exibe a imagem usando matplotlib.
    """
    # Carregar a imagem
    img = cv2.imread(image_path)
    # Converter de BGR (padrão OpenCV) para RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Exibir a imagem
    plt.imshow(img)
    plt.axis('off')  # Desliga os eixos
    plt.show()

def display_detections(results_folder):
    """
    Exibe todas as imagens de detecção na pasta de resultados.
    """
    # Obter todos os arquivos de imagem na pasta de resultados
    for result_image in os.listdir(results_folder):
        if result_image.endswith(".jpg") or result_image.endswith(".png"):
            image_path = os.path.join(results_folder, result_image)
            print(f"Exibindo: {image_path}")
            display_image(image_path)

# Exemplo de uso:
# Defina o caminho para a pasta de detecções
output_path = "outputs/detections"
display_detections(output_path)
