import cv2

def preprocess_image(image_path, target_size=(640, 640)):
    """
    Pré-processa a imagem para o formato esperado pelo modelo YOLO.
    
    Args:
        image_path (str): Caminho para a imagem de entrada.
        target_size (tuple): Tamanho desejado para a imagem (largura, altura).
    
    Returns:
        image: Imagem pré-processada.
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    
    # Redimensiona a imagem
    image = cv2.resize(image, target_size)
    
    return image