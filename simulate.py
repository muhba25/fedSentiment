"""
simulate.py
-----------
Script simulasi Federated Learning di 1 mesin (tanpa cluster fisik).
Berguna untuk testing cepat sebelum deploy ke multinode cluster.

Menggunakan flwr.simulation (VirtualClientEngine) yang menjalankan
semua klien sebagai proses terpisah secara efisien di 1 mesin.

Penggunaan:
  python simulate.py                     # IMDB dataset
  python simulate.py --dummy             # Dataset dummy (cepat)
  python simulate.py --rounds 5 --dummy  # 5 rounds dengan dummy
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters
from loguru import logger

# Import komponen project
from model.cnn_text import get_model
from client.client import SentimentClient
from server.strategy import FedSentimentStrategy
from utils.metrics import plot_fl_history, print_fl_report


def client_fn(cid: str, config: dict, use_dummy: bool) -> fl.client.Client:
    """
    Factory function untuk membuat klien FL.
    Dipanggil oleh Flower simulation engine untuk setiap klien.
    """
    node_id = int(cid)
    return SentimentClient(
        node_id=node_id,
        config=config,
        use_dummy=use_dummy,
    )


def run_simulation(config: dict, use_dummy: bool = False) -> None:
    """
    Jalankan simulasi FL di 1 mesin menggunakan Flower VirtualClientEngine.
    """
    server_cfg = config["server"]
    save_dir   = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Inisialisasi model untuk parameter awal
    model = get_model(config)
    initial_params = ndarrays_to_parameters(model.get_parameters())

    # Bangun strategi
    strategy = FedSentimentStrategy(
        config=config,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=server_cfg["min_fit_clients"],
        min_evaluate_clients=server_cfg["min_evaluate_clients"],
        min_available_clients=server_cfg["min_available_clients"],
        initial_parameters=initial_params,
        on_fit_config_fn=lambda rnd: {
            "current_round": str(rnd),
            "local_epochs": str(config["training"]["local_epochs"]),
        },
    )

    logger.info("=" * 60)
    logger.info("  FedSentiment — MODE SIMULASI")
    logger.info(f"  Rounds  : {server_cfg['num_rounds']}")
    logger.info(f"  Klien   : {server_cfg['min_available_clients']}")
    logger.info(f"  Dataset : {'dummy' if use_dummy else 'IMDB'}")
    logger.info(f"  Device  : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("=" * 60)

    # Jalankan simulasi
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, config, use_dummy),
        num_clients=server_cfg["min_available_clients"],
        config=fl.server.ServerConfig(num_rounds=server_cfg["num_rounds"]),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.0,  # Set > 0 jika GPU tersedia dan ingin dibagi
        },
    )

    # Tampilkan ringkasan
    strategy.print_summary()

    # Plot hasil
    if strategy.history:
        print_fl_report(strategy.history)
        plot_fl_history(
            strategy.history,
            save_path=f"{save_dir}/fl_training_curve.png",
        )
        logger.info(f"\nSemua hasil tersimpan di: {save_dir}/")

    return history


def main():
    parser = argparse.ArgumentParser(description="FedSentiment Simulasi FL")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dummy", action="store_true",
                        help="Gunakan dataset dummy (lebih cepat)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override jumlah FL rounds")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override rounds jika diberikan
    if args.rounds:
        config["server"]["num_rounds"] = args.rounds
        logger.info(f"Rounds dioverride: {args.rounds}")

    # Setup logging
    logger.add(
        f"{config['logging']['save_dir']}/simulate.log",
        level=config["logging"]["level"],
    )

    # Jalankan simulasi
    run_simulation(config=config, use_dummy=args.dummy)


if __name__ == "__main__":
    main()
