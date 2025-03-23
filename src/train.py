import argparse #para lidar com argumentos de linha de comando
import yaml #carrega arquivos de configuração no formato YAML
from pathlib import Path #manipulação de caminhos de arquivos
import torch 
from yolov5 import train #
import sys
import os

def load_hyperparameters(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(opt):
    # Carregar hiperparâmetros do arquivo de configuração
    hyp = load_hyperparameters(opt.hyp) if opt.hyp else None
    
    # Configurar argumentos para o treinamento
    args = {
        'weights': opt.weights,
        'cfg': opt.cfg,
        'data': opt.data,
        'epochs': opt.epochs,
        'batch_size': opt.batch_size,
        'img_size': opt.img_size,
        'hyp': hyp,
        'project': opt.project,
        'name': opt.name,
        'exist_ok': opt.exist_ok
    }
    
    # Iniciar treinamento
    train.run(**args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Caminho para os pesos do modelo')
    parser.add_argument('--cfg', type=str, default='', help='Caminho para o arquivo de configuração do modelo')
    parser.add_argument('--data', type=str, default='data/custom_data.yaml', help='Caminho para o arquivo de dados')
    parser.add_argument('--hyp', type=str, default='', help='Caminho para o arquivo de hiperparâmetros')
    parser.add_argument('--epochs', type=int, default=300, help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do batch de treinamento')
    parser.add_argument('--img_size', type=int, default=640, help='Tamanho das imagens de entrada')
    parser.add_argument('--project', type=str, default='runs/train', help='Diretório do projeto para salvar resultados')
    parser.add_argument('--name', type=str, default='exp', help='Nome da execução do experimento')
    parser.add_argument('--exist_ok', action='store_true', help='Permitir que o diretório do projeto exista')

    opt = parser.parse_args()
    main(opt)