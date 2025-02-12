import os

def predict_images(weights_path, source_path, output_path):
    """
    Faz previsões em novas imagens usando YOLOv5.
    """
    yolov5_path = "C:/caminho/para/o/repositorio/yolov5"  # Caminho do repositório YOLOv5
    os.system(f"python {yolov5_path}/detect.py --weights {weights_path} --img 416 --conf 0.4 --source {source_path} --save-txt --project {output_path}")

# Exemplo de uso
predict_images("yolov5/runs/train/exp/weights/best.pt", "data/raw", "outputs")
