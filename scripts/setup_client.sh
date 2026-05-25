#!/bin/bash
# ============================================================
# scripts/setup_client.sh
# Setup Node-2, Node-3, Node-4 sebagai Flower FL Client
# Penggunaan: bash scripts/setup_client.sh <node_id> <server_ip>
# Contoh    : bash scripts/setup_client.sh 0 192.168.1.10
# ============================================================

set -euo pipefail

# ── Argumen ──────────────────────────────────────────────
NODE_ID="${1:-0}"
SERVER_IP="${2:-192.168.1.10}"
SERVER_PORT="${3:-9091}"
SERVER_ADDRESS="${SERVER_IP}:${SERVER_PORT}"

echo "======================================================"
echo "  FedSentiment — Setup Client Node"
echo "  Node ID    : $NODE_ID"
echo "  Server     : $SERVER_ADDRESS"
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

# ── 2. Virtual environment ────────────────────────────────
echo "[2/7] Membuat Python virtual environment..."
python3 -m venv $HOME/fedsentiment_env
source $HOME/fedsentiment_env/bin/activate
pip install --upgrade pip setuptools wheel

# ── 3. Install dependencies ───────────────────────────────
echo "[3/7] Instalasi Python dependencies..."
pip install -r requirements.txt

# ── 4. Setup project ──────────────────────────────────────
echo "[4/7] Setup direktori project..."
PROJECT_DIR="$HOME/fedsentiment"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Buat direktori project di $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    touch "$PROJECT_DIR"/model/__init__.py
    touch "$PROJECT_DIR"/client/__init__.py
    touch "$PROJECT_DIR"/server/__init__.py
    touch "$PROJECT_DIR"/utils/__init__.py
fi

# ── 5. Update config.yaml untuk klien ini ────────────────
echo "[5/7] Update konfigurasi klien..."

# Jika file config belum ada, buat dari template
CONFIG_FILE="$PROJECT_DIR/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    # Update server address dan node_id di config
    sed -i "s|server_address:.*|server_address: \"$SERVER_ADDRESS\"|" "$CONFIG_FILE"
    sed -i "s|node_id:.*|node_id: $NODE_ID|" "$CONFIG_FILE"
    echo "config.yaml diperbarui:"
    grep -A2 "^client:" "$CONFIG_FILE"
else
    echo "WARNING: config.yaml belum ada di $CONFIG_FILE"
    echo "Pastikan Anda menyalin config.yaml dari server ke direktori ini"
fi

# ── 6. Test koneksi ke server ─────────────────────────────
echo "[6/7] Test koneksi ke server $SERVER_IP..."

# Test ping
if ping -c 3 "$SERVER_IP" > /dev/null 2>&1; then
    echo "✓ Server dapat di-ping"
else
    echo "✗ WARNING: Tidak bisa ping server $SERVER_IP"
    echo "  Pastikan server sudah berjalan dan firewall terbuka"
fi

# Test port
if nc -z -w5 "$SERVER_IP" "$SERVER_PORT" 2>/dev/null; then
    echo "✓ Port $SERVER_PORT terbuka di $SERVER_IP"
else
    echo "✗ WARNING: Port $SERVER_PORT tidak terbuka di $SERVER_IP"
    echo "  Jalankan server terlebih dahulu"
fi

# ── 7. Verifikasi ─────────────────────────────────────────
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

# ── Buat script run shortcut ──────────────────────────────
cat > "$PROJECT_DIR/run_client.sh" << RUNEOF
#!/bin/bash
# Shortcut untuk menjalankan klien
source $HOME/fedsentiment_env/bin/activate
cd $PROJECT_DIR
python client/client.py \\
    --node_id $NODE_ID \\
    --server $SERVER_ADDRESS \\
    --config config.yaml \\
    "\$@"
RUNEOF
chmod +x "$PROJECT_DIR/run_client.sh"

echo ""
echo "======================================================"
echo "  Setup Client Node-$NODE_ID SELESAI!"
echo "======================================================"
echo ""
echo "IP Node ini:"
ip addr show | grep "inet " | grep -v "127.0.0.1" | awk '{print "  " $2}'
echo ""
echo "Cara menjalankan klien:"
echo "  cd $PROJECT_DIR"
echo "  source $HOME/fedsentiment_env/bin/activate"
echo ""
echo "  # Opsi 1: Gunakan dataset local (perlu copy dataset ke klien)"
echo "  python client/client.py --node_id $NODE_ID --server $SERVER_ADDRESS --config config.yaml"
echo ""
echo "  # Opsi 2: Gunakan dataset IMDB (perlu internet)"
echo "  python client/client.py --node_id $NODE_ID --server $SERVER_ADDRESS --config config.yaml"
echo ""
echo "  # Opsi 3: Gunakan dataset dummy (tanpa internet)"
echo "  python client/client.py --node_id $NODE_ID --server $SERVER_ADDRESS --dummy"
echo ""
echo "  # Opsi 4: Shortcut"
echo "  bash $PROJECT_DIR/run_client.sh"
echo "======================================================"
