import cv2
from ultralytics import YOLO

# Carregando o modelo treinado
model = YOLO("yolov8n.pt")  # Ou 'runs/detect/train/weights/best.pt' se estiver em outro local
model.to('cpu')
# Inicializando a webcam
cap = cv2.VideoCapture(0)  # 0 = webcam padrão

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fazendo a predição
    results = model(frame)[0]
    
    # Desenhando os resultados na imagem
    annotated_frame = results.plot()

    # Exibindo
    cv2.imshow("Detecção com YOLOv8", annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrando
cap.release()
cv2.destroyAllWindows()
