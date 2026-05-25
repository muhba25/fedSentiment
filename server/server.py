"""
server/server.py
----------------
Flower Server (SuperLink) untuk FedSentiment.

Cara penggunaan:
  # Di Node-1 (server):
  python server/server.py

Server ini bertanggung jawab untuk:
  1. Menunggu koneksi dari min_available_clients klien
  2. Mendistribusikan global model ke klien (konfigurasi fit)
  3. Mengumpulkan parameter lokal dan mengagregasi (FedAvg)
  4. Mengevaluasi global model di klien
  5. Mengulangi siklus selama num_rounds round
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters
from loguru import logger

from model.cnn_text import get_model
from server.strategy import FedSentimentStrategy


def build_strategy(config: dict) -> FedSentimentStrategy:
    """
    Bangun strategi FL dari konfigurasi.
    """
    server_cfg = config["server"]

    # Inisialisasi model untuk mendapatkan parameter awal
    model = get_model(config)
    initial_params = ndarrays_to_parameters(model.get_parameters())

    strategy = FedSentimentStrategy(
        config=config,
        # Parameter FedAvg standar
        fraction_fit=1.0,          # Gunakan 100% klien yang tersedia untuk fit
        fraction_evaluate=1.0,     # Gunakan 100% klien untuk evaluasi
        min_fit_clients=server_cfg["min_fit_clients"],
        min_evaluate_clients=server_cfg["min_evaluate_clients"],
        min_available_clients=server_cfg["min_available_clients"],
        initial_parameters=initial_params,
        # Callback evaluasi (opsional: evaluasi di server dengan data terpusat)
        evaluate_fn=None,
        on_fit_config_fn=lambda rnd: {
            "current_round": str(rnd),
            "local_epochs": str(config["training"]["local_epochs"]),
        },
        on_evaluate_config_fn=lambda rnd: {
            "current_round": str(rnd),
        },
    )

    return strategy


def main():
    parser = argparse.ArgumentParser(description="FedSentiment Flower Server")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path ke file konfigurasi"
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="Override host server"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override port server"
    )
    args = parser.parse_args()

    # Load konfigurasi
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Tentukan address
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]
    server_address = f"{host}:{port}"

    # Setup logging
    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    logger.add(
        f"{save_dir}/server.log",
        level=config["logging"]["level"],
        rotation="10 MB",
    )

    logger.info("=" * 60)
    logger.info("  FedSentiment — Flower FL Server")
    logger.info(f"  Address : {server_address}")
    logger.info(f"  Rounds  : {config['server']['num_rounds']}")
    logger.info(f"  Clients : {config['server']['min_available_clients']}")
    logger.info(f"  Model   : TextCNN (PyTorch)")
    logger.info(f"  Dataset : IMDB Sentiment")
    logger.info("=" * 60)
    logger.info("Menunggu koneksi klien...")

    # Bangun strategi
    strategy = build_strategy(config)

    # Konfigurasi server
    server_config = ServerConfig(
        num_rounds=config["server"]["num_rounds"],
        round_timeout=None,  # Tidak ada timeout per round
    )

    # Mulai server
    start_time = time.time()
    history = fl.server.start_server(
        server_address=server_address,
        config=server_config,
        strategy=strategy,
    )
    elapsed = time.time() - start_time

    # Cetak ringkasan
    strategy.print_summary()
    logger.info(f"\nTotal waktu FL: {elapsed:.1f} detik ({elapsed/60:.1f} menit)")
    logger.info(f"Hasil tersimpan di: {save_dir}/")

    # Cetak metrik terakhir dari history Flower
    if history.losses_distributed:
        final_round, final_loss = history.losses_distributed[-1]
        logger.info(f"Loss akhir (round {final_round}): {final_loss:.4f}")

    if history.metrics_distributed.get("accuracy"):
        accs = history.metrics_distributed["accuracy"]
        final_round, final_acc = accs[-1]
        logger.info(f"Akurasi akhir (round {final_round}): {final_acc:.4f}")


if __name__ == "__main__":
    main()
