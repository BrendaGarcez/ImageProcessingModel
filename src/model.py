from ultralytics import YOLO

def load_model(model_path="yolov8n.pt"):
    """
    Carrega o modelo YOLO pré-treinado ou treinado.
    
    Args:
        model_path (str): Caminho para o arquivo de pesos do modelo (.pt).
    
    Returns:
        model: Modelo YOLO carregado.
    """
    model = YOLO(model_path)
    return model