from model import load_model

def predict(image_path, model_path="runs/train/exp/weights/best.pt", conf=0.5):
    """
    Faz a detecção de componentes em uma imagem usando o modelo YOLO.
    
    Args:
        image_path (str): Caminho para a imagem de entrada.
        model_path (str): Caminho para o arquivo de pesos do modelo (.pt).
        conf (float): Limiar de confiança para as detecções.
    
    Returns:
        results: Resultados da detecção.
    """
    # Carrega o modelo
    model = load_model(model_path)
    
    # Faz a predição
    results = model.predict(source=image_path, conf=conf, save=True)
    
    return results