#!/bin/bash

# Define o diretório da virtual environment
VENV_DIR="myenv"

# Verifica se o diretório da virtual environment existe
if [ ! -d "$VENV_DIR" ]; then
    echo "Criando a virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Ativa a virtual environment
source "$VENV_DIR/bin/activate"

# Atualiza o pip
pip install --upgrade pip

# Instala as dependências necessárias
pip install gymnasium stable-baselines3 numpy selenium

echo "Instalando PyTorch (cu118)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Ambiente virtual criado e pacotes instalados com sucesso!"
