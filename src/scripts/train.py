from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  # Modelo pré-treinado
    model.train(data="src/dataset.yaml", epochs=100, workers=2)
    results = model.train(
        data="src/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device="cpu",  # GPU
        name="placas_v1"
    )
    return results

if __name__ == "__main__":
    train()