#carrega um modelo YOLO usando a biblioteca Ultralytics.
from ultralytics import YOLO

def load_model(model_path="models/yolov8n.pt"):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        raise