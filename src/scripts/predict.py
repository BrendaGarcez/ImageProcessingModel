import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse

def predict(image_path, model_path="models/best.pt", output_dir="outputs/"):
     # Criar diretório de saída se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Carregar modelo
        model = YOLO(model_path)
        
        # Fazer previsão
        results = model.predict(image_path, conf=0.5)
        
        # Preparar nomes de arquivo de saída
        stem = Path(image_path).stem
        output_img = f"{output_dir}/{stem}_pred.jpg"
        output_txt = f"{output_dir}/{stem}_pred.txt"
        
        # Salvar resultados
        results[0].save(output_img)
        results[0].save_txt(output_txt)
        
        print(f"Resultados salvos em:\n- {output_img}\n- {output_txt}")
        return results
        
    except Exception as e:
        print(f"Erro durante a predição: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, 
                       help="Caminho para a imagem de entrada")
    parser.add_argument("--model", type=str, default="models/best.pt",
                       help="Caminho para o modelo YOLO")
    args = parser.parse_args()
    
    predict(image_path=args.image, model_path=args.model)