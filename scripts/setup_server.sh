#!/bin/bash
# ============================================================
# scripts/setup_server.sh
# Setup Node-1 sebagai Flower FL Server
# Jalankan: bash scripts/setup_server.sh
# ============================================================

set -euo pipefail

echo "======================================================"
echo "  FedSentiment — Setup Node-1 (FL Server)"
echo "  Ubuntu 22.04 · Flower Framework"
echo "======================================================"

# ── 1. Update sistem ──────────────────────────────────────
echo "[1/7] Update sistem..."
sudo apt-get update -y
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    wget \
    htop \
    net-tools

# ── 2. Buat virtual environment ──────────────────────────
echo "[2/7] Membuat Python virtual environment..."
python3 -m venv $HOME/fedsentiment_env
source $HOME/fedsentiment_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ── 3. Install dependencies ───────────────────────────────
echo "[3/7] Instalasi Python dependencies..."
pip install -r requirements.txt

echo "Dependencies terinstall:"
pip show flwr torch | grep -E "Name|Version"

# ── 4. Clone/setup project ────────────────────────────────
echo "[4/7] Setup project FedSentiment..."
PROJECT_DIR="$HOME/fedsentiment"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Buat direktori project di $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
fi


# ── 5. Verifikasi instalasi ───────────────────────────────
echo "[7/7] Verifikasi instalasi..."

source $HOME/fedsentiment_env/bin/activate
python3 -c "
import flwr
import torch
print(f'Flower  : {flwr.__version__}')
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
print('✓ Semua library OK')
"

# Tampilkan IP node ini
echo ""
echo "======================================================"
echo "  Setup Node-1 (Server) SELESAI!"
echo "======================================================"
echo "IP Node ini:"
ip addr show | grep "inet " | grep -v "127.0.0.1" | awk '{print "  " $2}'
echo ""
echo "Langkah selanjutnya:"
echo "  1. Copy semua file Python ke $PROJECT_DIR"
echo "  2. Edit config.yaml: server.host = 0.0.0.0"
echo "  3. Jalankan server:"
echo "     source $HOME/fedsentiment_env/bin/activate"
echo "     cd $PROJECT_DIR"
echo "     python server/server.py --config config.yaml "
echo "======================================================"
