import os

def train_model(data_path, epochs=10):
    """
    Treina o modelo YOLOv5 com os dados fornecidos.
    """
    yolov5_path = "C:/caminho/para/o/repositorio/yolov5"  # Caminho do repositório YOLOv5
    os.system(f"python {yolov5_path}/train.py --img 416 --batch 16 --epochs {epochs} --data {data_path} --weights yolov5s.pt")

# Exemplo de uso
train_model("data/labeled.yaml", epochs=50)
