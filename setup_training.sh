#!/bin/bash

# Setup script for TitanLLaMA training environment

set -e

echo "Setting up TitanLLaMA training environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install titans-pytorch dependencies
echo "Installing titans-pytorch dependencies..."
pip install einops
pip install axial-positional-embedding
pip install rotary-embedding-torch
pip install x-transformers

# Try to install hyper-connections (might need to be installed from source)
pip install git+https://github.com/lucidrains/hyper-connections.git || echo "Warning: Could not install hyper-connections"

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements_training.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p titan_llama_checkpoints
mkdir -p logs
mkdir -p data

# Check if titans-pytorch is properly set up
echo "Checking titans-pytorch setup..."
python -c "
try:
    from titans_pytorch import MemoryAsContextTransformer, NeuralMemory, MemoryMLP
    print('✓ titans-pytorch imports successful')
except ImportError as e:
    print(f'✗ titans-pytorch import error: {e}')
    print('Make sure titans-pytorch is in your Python path')
"

# Check if our TitanLLaMA implementation works
echo "Checking TitanLLaMA implementation..."
python -c "
try:
    from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM
    print('✓ TitanLLaMA imports successful')
    
    # Test model creation
    config = TitanLLaMAConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=4,
        neural_memory_layers=(2,)
    )
    model = TitanLLaMAForCausalLM(config)
    print(f'✓ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters')
except Exception as e:
    print(f'✗ TitanLLaMA error: {e}')
"

# Set up environment variables
echo "Setting up environment variables..."
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline  # Set to online if you want to use wandb

echo ""
echo "✓ Setup complete!"
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python run_training.py --help  # See all options"
echo ""
echo "Quick start (debug mode):"
echo "  python run_training.py --debug --no_wandb"
echo ""
echo "Full training (1B tokens):"
echo "  python run_training.py --total_tokens 1000000000 --batch_size 8 --micro_batch_size 2"
echo ""

# Make run scripts executable
chmod +x run_training.py
chmod +x setup_training.sh

echo "Environment setup complete!"