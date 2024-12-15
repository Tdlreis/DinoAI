@echo off
set VENV_DIR=myenv

REM Cria a virtual environment
if not exist %VENV_DIR% (
    echo Criando a virtual environment...
    python -m venv %VENV_DIR%
)

REM Ativa a virtual environment
call %VENV_DIR%\Scripts\activate

REM Atualiza o pip
python.exe -m pip install --upgrade pip

REM Instala as dependências necessárias
pip install gymnasium stable-baselines3 numpy selenium tensorboard

echo Instalando PyTorch (cu118)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Ambiente virtual criado e pacotes instalados com sucesso!