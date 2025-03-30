import matplotlib.pyplot as plt
from ultralytics import YOLO

def plot_results(model_path="models/best.pt"):
    model = YOLO(model_path)
    metrics = model.val()  # Gera gráficos automaticamente
    plt.show()  # Mostra curvas de precisão/recall

if __name__ == "__main__":
    plot_results()