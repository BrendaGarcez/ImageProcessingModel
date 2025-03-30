import cv2
from ultralytics import YOLO


def predict(image_path, model_path="models/best.pt"):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.5)
    
    # Salvar resultados
    results[0].save("output.jpg")  # Imagem com bboxes
    results[0].save_txt("output.txt")  # Coordenadas dos componentes
    
    return results

if __name__ == "__main__":
    predict("src/images/placa_test.jpg")