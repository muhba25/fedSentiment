"""
client/client.py  — FedSentiment (fixed)

FIX 1: NumPyClient menggantikan raw Client
  → Menghindari flwr/compat/ traceback di Flower >= 1.8

FIX 2: DataLoader num_workers=0, pin_memory=False
  → Menghindari "DataLoader worker exited unexpectedly"
    akibat fork di dalam Flower subprocess
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
import flwr as fl
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader
from loguru import logger
from typing import Dict, Tuple

from model.cnn_text import get_model
from client.data_loader import get_data_loaders


def train(model, loader, optimizer, criterion, epochs, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for epoch in range(epochs):
        ep_loss = 0.0
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item() * ids.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += ids.size(0)
        total_loss += ep_loss / len(loader.dataset)
        logger.debug(f"  Epoch {epoch+1}/{epochs} loss={ep_loss/len(loader.dataset):.4f}")
    return total_loss / epochs, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * ids.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += ids.size(0)
    return total_loss / total, correct / total


class SentimentClient(NumPyClient):
    """
    NumPyClient untuk FedSentiment.
    get_parameters / fit / evaluate menerima & mengembalikan List[np.ndarray].
    """

    def __init__(self, node_id: int, config: dict, use_dummy: bool = False):
        self.node_id   = node_id
        self.config    = config
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = get_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
        )

        logger.info(f"[Node {node_id}] Device: {self.device} | Flower {fl.__version__}")
        self.train_loader, self.test_loader = get_data_loaders(
            node_id=node_id, config=config, use_dummy=use_dummy,
        )
        logger.info(f"[Node {node_id}] Siap.")

    def get_parameters(self, config: Dict) -> NDArrays:
        return self.model.get_parameters()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.model.set_parameters(parameters)
        local_epochs  = int(config.get("local_epochs",  self.config["training"]["local_epochs"]))
        current_round = int(config.get("current_round", 0))
        logger.info(f"[Node {self.node_id}] Round {current_round} | fit {local_epochs} epoch")
        loss, acc = train(self.model, self.train_loader, self.optimizer,
                          self.criterion, local_epochs, self.device)
        logger.info(f"[Node {self.node_id}] Round {current_round} | loss={loss:.4f} acc={acc:.4f}")
        return (
            self.model.get_parameters(),
            len(self.train_loader.dataset),
            {"train_loss": float(loss), "train_accuracy": float(acc), "node_id": float(self.node_id)},
        )

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.model.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.criterion, self.device)
        logger.info(f"[Node {self.node_id}] Evaluate | loss={loss:.4f} acc={acc:.4f}")
        return (
            float(loss),
            len(self.test_loader.dataset),
            {"accuracy": float(acc), "node_id": float(self.node_id)},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_id", type=int, required=True)
    parser.add_argument("--config",  type=str, default="config.yaml")
    parser.add_argument("--dummy",   action="store_true")
    parser.add_argument("--server",  type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    server_address = args.server or config["client"]["server_address"]
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    logger.add(
        f"{config['logging']['save_dir']}/client_{args.node_id}.log",
        level=config["logging"]["level"], rotation="10 MB",
    )

    logger.info("=" * 60)
    logger.info(f"FedSentiment Client — Node {args.node_id}")
    logger.info(f"Server : {server_address}")
    logger.info(f"Dataset: {'dummy' if args.dummy else 'IMDB'}")
    logger.info("=" * 60)

    client = SentimentClient(node_id=args.node_id, config=config, use_dummy=args.dummy)

    # .to_client() = konversi NumPyClient → Client, wajib di Flower >= 1.8
    fl.client.start_client(server_address=server_address, client=client.to_client())
    logger.info(f"[Node {args.node_id}] FL selesai.")


if __name__ == "__main__":
    main()
