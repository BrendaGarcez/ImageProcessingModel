from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  # Modelo pré-treinado
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device="0",  # GPU
        name="placas_v1"
    )
    return results

if __name__ == "__main__":
    train()