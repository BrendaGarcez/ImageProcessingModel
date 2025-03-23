import cv2

def visualize_results(image_path, results):
    """
    Exibe as bounding boxes e rótulos das detecções na imagem.
    
    Args:
        image_path (str): Caminho para a imagem de entrada.
        results: Resultados da detecção retornados pelo modelo YOLO.
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    
    # Itera sobre as detecções
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Coordenadas da bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Rótulo e confiança
            label = f"{result.names[int(box.cls)]} {box.conf:.2f}"
            # Desenha a bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Adiciona o rótulo
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Exibe a imagem
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()