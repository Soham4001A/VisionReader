#!/bin/bash

echo "🚀 Starting Project Setup..."

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p raw_data synthetic_data processed_data/train processed_data/val models

# Create a Python virtual environment
echo "🐍 Creating Python virtual environment 'venv'..."
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📦 Installing required Python packages..."
pip install torch torchvision torchaudio
pip install opencv-python mediapipe numpy scikit-learn matplotlib tqdm

echo "✅ Setup complete! To activate the environment, run: source venv/bin/activate"