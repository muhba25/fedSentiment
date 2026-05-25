# PANDUAN DEPLOYMENT LENGKAP — FedSentiment

## Flower Federated Learning · 3-Node Ubuntu 22.04 Cluster

---

## RINGKASAN ARSITEKTUR

```
┌─────────────────────────────────────────────┐
│           FLOWER MULTINODE CLUSTER          │
│                                             │
│  Node-1 (192.168.1.10) — FL Server          │
│  ┌──────────────────────────────────┐       │
│  │  flower.server.start_server()    │       │
│  │  Strategy: FedAvg                │       │
│  │  Port: 9091 (gRPC)               │       │
│  └─────────────┬────────────────────┘       │
│                │ Global Model               │
│       ┌────────┼────────┐                   │
│       ▼        ▼        ▼                   │
│  Node-2     Node-3     Node-4               │
│  Client-0   Client-1   Client-2             │
│  IMDB 1/3   IMDB 2/3   IMDB 3/3             │
│  TextCNN    TextCNN    TextCNN              │
└─────────────────────────────────────────────┘
```

---

## STEP 1 — PERSIAPAN INFRASTRUKTUR

### 1.1 Konfigurasi Jaringan

Pastikan semua node berada dalam satu jaringan lokal (LAN) dan saling dapat berkomunikasi.

**Di semua node, edit /etc/hosts:**

```bash
sudo tee -a /etc/hosts << 'EOF'
192.168.1.10   fl-server
192.168.1.11   fl-client-0
192.168.1.12   fl-client-1
192.168.1.13   fl-client-2
EOF
```

**Test konektivitas:**

```bash
# Dari Node-2,3,4: ping ke server
ping -c 3 192.168.1.10

# Dari Node-1: ping ke semua klien
ping -c 3 192.168.1.11
ping -c 3 192.168.1.12
ping -c 3 192.168.1.13
```

### 1.2 Sinkronisasi Waktu

```bash
sudo apt-get install -y chrony
sudo systemctl start chrony
sudo systemctl enable chrony
timedatectl status
```

---

## STEP 2 — SETUP NODE-1 (SERVER)

```bash
# Di Node-1 (192.168.1.10)
cd $HOME
git clone <repo> fedsentiment   # atau scp dari mesin development
cd fedsentiment

# Aktifkan virtualenv
source $HOME/fedsentiment_env/bin/activate

# Jalankan script setup
chmod +x scripts/setup_server.sh
bash scripts/setup_server.sh
```

---

## STEP 3 — SETUP NODE-2, 3, 4 (CLIENT)

**Di setiap node klien, jalankan:**

```bash
# Node-2 (node_id=0)
bash $HOME/fedsentiment/scripts/setup_client.sh 0 192.168.1.10

# Node-3 (node_id=1)
bash $HOME/fedsentiment/scripts/setup_client.sh 1 192.168.1.10

# Node-4 (node_id=2)
bash $HOME/fedsentiment/scripts/setup_client.sh 2 192.168.1.10
```

**Copy file project ke setiap klien:**

```bash
# Dari mesin development atau Node-1:
scp -r ~/fedsentiment/ ubuntu@192.168.1.11:~/
scp -r ~/fedsentiment/ ubuntu@192.168.1.12:~/
scp -r ~/fedsentiment/ ubuntu@192.168.1.13:~/
```

---

## STEP 4 — KONFIGURASI

**Edit config.yaml di tiap node:**

**Node-1 (server):**

```yaml
server:
  host: "0.0.0.0"
  port: 9091
  num_rounds: 10
  min_fit_clients: 3
  min_evaluate_clients: 3
  min_available_clients: 3
```

**Node-2, 3, 4 (client) — edit server address:**

```yaml
client:
  server_address: "192.168.1.10:9091"
  node_id: 0 # Ganti: 0 untuk Node-2, 1 untuk Node-3, 2 untuk Node-4
```

---

## STEP 5 — MENJALANKAN EKSPERIMEN

### Urutan Penting: Server Dulu, Baru Client

**Terminal 1 — Node-1 (Server):**

```bash
source $HOME/fedsentiment_env/bin/activate
cd $HOME/fedsentiment
python server/server.py --config config.yaml
```

Output yang diharapkan:

```
INFO  FedSentiment — Flower FL Server
INFO  Address : 0.0.0.0:9091
INFO  Rounds  : 10
INFO  Menunggu koneksi klien...
```

**Terminal 2 — Node-2 (Client-0):**

```bash
source $HOME/fedsentiment_env/bin/activate
cd $HOME/fedsentiment
python client/client.py --node_id 0 --config config.yaml
```

**Terminal 3 — Node-3 (Client-1):**

```bash
source $HOME/fedsentiment_env/bin/activate
cd $HOME/fedsentiment
python client/client.py --node_id 1 --config config.yaml
```

**Terminal 4 — Node-4 (Client-2):**

```bash
source $HOME/fedsentiment_env/bin/activate
cd $HOME/fedsentiment
python client/client.py --node_id 2 --config config.yaml
```

### Mode Simulasi Cepat (tanpa cluster fisik):

```bash
python simulate.py --dummy --rounds 5
```

---

## STEP 6 — MONITORING

### Memonitor Log Server (Node-1):

```bash
tail -f results/server.log
```

### Memonitor Log Klien:

```bash
# Di masing-masing node klien:
tail -f results/client_0.log
tail -f results/client_1.log
tail -f results/client_2.log
```

### Output yang Diharapkan per Round:

```
==================================================
  FL Round 3/10 — AGGREGATION
==================================================
  Client 0: loss=0.4821 | acc=0.7531 | samples=8334
  Client 1: loss=0.4903 | acc=0.7448 | samples=8333
  Client 2: loss=0.4765 | acc=0.7612 | samples=8333
  Agregat: loss=0.4830 | acc=0.7530

  ── Evaluasi Round 3 ──
  Eval loss    : 0.4710
  Eval accuracy: 0.7591
```

---

## STEP 7 — ANALISIS HASIL

Setelah FL selesai, analisis hasil:

```bash
# Lihat ringkasan
cat results/fl_history.json

# Buat plot kurva training
python utils/metrics.py results/fl_history.json
# Output: results/fl_training_curve.png
```

---

## TROUBLESHOOTING

### Error: "Connection refused" di klien

```bash
# Pastikan server sudah berjalan
netstat -tlnp | grep 9091

# Cek firewall di server
sudo ufw status
sudo ufw allow 9091/tcp
```

### Error: "Not enough clients connected"

```bash
# Pastikan semua klien terhubung sebelum round dimulai
# Ubah di config.yaml:
server:
  min_available_clients: 2  # Kurangi jika 1 klien bermasalah
```

### Error: "CUDA out of memory"

```bash
# Kurangi batch size di config.yaml:
training:
  batch_size: 32  # Default 64, kurangi jika OOM
```

### Dataset IMDB gagal diunduh

```bash
# Gunakan dataset dummy:
python client/client.py --node_id 0 --dummy
# atau di simulate.py:
python simulate.py --dummy
```

---

## REFERENSI

- Flower Framework: https://flower.ai/docs/
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP.
- IMDB Dataset: https://huggingface.co/datasets/imdb
