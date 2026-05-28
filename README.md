# FedSentiment — Federated Learning Sentiment Analysis

## Mini Project S2 · Flower Framework · Multinode Cluster

### Deskripsi Project

FedSentiment adalah implementasi **Federated Learning** untuk klasifikasi sentimen teks (positif/negatif) menggunakan dataset IMDB Movie Reviews. Project ini menggunakan **Flower (flwr)** framework dengan arsitektur **3-node Ubuntu 22.04** dalam jaringan lokal, model **CNN-Text (PyTorch)**, dan algoritma agregasi **FedAvg**.

---

## Fedsentiment Topology
![Architectur Topology Fedsentiment](https://github.com/muhba25/fedSentiment/blob/master/src/img/federated_learning_architecture.svg)

## Struktur Project

```
fedsentiment/
├── README.md
├── requirements.txt
├── config.yaml                  # Konfigurasi global
├── server/
│   ├── server.py                # Flower SuperLink server
│   └── strategy.py              # Custom FedAvg strategy
├── client/
│   ├── client.py                # Flower client (dijalankan di tiap node)
│   └── data_loader.py           # Loader & partisi dataset IMDB
├── model/
│   └── cnn_text.py              # Definisi model CNN-Text
├── utils/
│   └── metrics.py               # Fungsi evaluasi
└── scripts/
    ├── setup_server.sh          # Setup Node-1 (server)
    ├── setup_client.sh          # Setup Node-2,3,4 (client)
    └── run_experiment.sh        # Jalankan eksperimen FL
```

---

## Arsitektur Cluster

| Role        | Hostname    | IP           | OS           |
| ----------- | ----------- | ------------ | ------------ |
| FL Server   | fl-server   | 192.168.1.10 | Ubuntu 22.04 |
| FL Client 0 | fl-client-0 | 192.168.1.11 | Ubuntu 22.04 |
| FL Client 1 | fl-client-1 | 192.168.1.12 | Ubuntu 22.04 |
| FL Client 2 | fl-client-2 | 192.168.1.13 | Ubuntu 22.04 |

> Catatan: Jika hanya tersedia 3 node, jalankan server + 1 client di node fl-server.

---

## Fedsentimen Flowchart
![Flowchart Fedsentiment](https://github.com/muhba25/fedSentiment/blob/master/src/img/fl_round_flowchart.svg)

## Spesifikasi Teknis

- **Framework FL**: Flower (flwr) >= 1.8
- **ML Framework**: PyTorch >= 2.0
- **Model**: CNN untuk klasifikasi teks
- **Dataset**: IMDB Movie Reviews (50.000 sampel, 25k train + 25k test)
- **Algoritma Agregasi**: FedAvg (McMahan et al., 2017)
- **FL Rounds**: 10 rounds
- **Local Epochs per Round**: 5 epochs
- **Jumlah Klien**: 3 klien simultan
