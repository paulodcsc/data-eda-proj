#!/bin/bash

echo "=== Setup do Projeto Python ==="
echo "Qual é o seu sistema operacional?"
echo "1) Linux / macOS"
echo "2) Windows (Git Bash / WSL)"
read -p "Escolha (1 ou 2): " so

VENV_DIR=".venv"

echo "-> Criando ambiente virtual..."
python3 -m venv $VENV_DIR 2>/dev/null || python -m venv $VENV_DIR

if [ $? -ne 0 ]; then
    echo "Erro ao criar o ambiente virtual. Verifique se o Python está instalado."
    exit 1
fi

if [ "$so" == "1" ]; then
    echo "-> Ativando venv (Linux/macOS)..."
    source "$VENV_DIR/bin/activate"

elif [ "$so" == "2" ]; then
    echo "-> Ativando venv (Windows)..."
    source "$VENV_DIR/Scripts/activate"
else
    echo "Opção inválida!"
    exit 1
fi

if [ -f "requirements.txt" ]; then
    echo "-> Instalando pacotes do requirements.txt..."
    pip install -r requirements.txt
else
    echo "Arquivo requirements.txt não encontrado!"
fi

echo "=== Setup concluído com sucesso! ==="
echo "Ambiente virtual ativado."
