"""
server/strategy.py
------------------
Custom FedAvg Strategy untuk FedSentiment.

Extends flwr.server.strategy.FedAvg dengan:
  - Logging per-round metrics
  - Weighted aggregation berdasarkan jumlah data klien
  - Menyimpan checkpoint model terbaik
  - Logging akurasi agregat per round
"""

import os
import json
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes, EvaluateRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger


class FedSentimentStrategy(FedAvg):
    """
    Strategi FL kustom berbasis FedAvg untuk FedSentiment.
    
    Fitur tambahan:
      - Mencetak ringkasan per round ke console
      - Menyimpan history metrik ke JSON
      - Menyimpan bobot model terbaik (berdasarkan akurasi evaluasi)
    """

    def __init__(self, config: dict, **kwargs):
        self.fl_config   = config
        self.save_dir    = config["logging"]["save_dir"]
        self.num_rounds  = config["server"]["num_rounds"]
        self.history     = {
            "rounds": [],
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
        }
        self.best_accuracy = 0.0
        self.best_params   = None
        self.current_round = 0

        os.makedirs(self.save_dir, exist_ok=True)

        # Inisialisasi FedAvg parent
        super().__init__(**kwargs)
        logger.info("FedSentimentStrategy diinisialisasi dengan FedAvg")

    # ── configure_fit ──────────────────────────────
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Tambahkan informasi round ke config yang dikirim ke klien."""
        self.current_round = server_round

        # Dapatkan config_fn dari FedAvg parent
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Tambahkan metadata round ke setiap klien
        updated_instructions = []
        for client, fit_ins in client_instructions:
            config = dict(fit_ins.config)
            config["current_round"] = str(server_round)
            config["local_epochs"]  = str(self.fl_config["training"]["local_epochs"])
            updated_fit_ins = fl.common.FitIns(
                parameters=fit_ins.parameters,
                config=config,
            )
            updated_instructions.append((client, updated_fit_ins))

        return updated_instructions

    # ── aggregate_fit ──────────────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agregasi hasil fit dari semua klien."""

        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} klien gagal!")
            for f in failures:
                logger.warning(f"  Failure: {f}")

        # Logging metrik per klien
        logger.info(f"\n{'='*50}")
        logger.info(f"  FL Round {server_round}/{self.num_rounds} — AGGREGATION")
        logger.info(f"{'='*50}")

        train_losses, train_accs = [], []
        for client, fit_res in results:
            metrics = fit_res.metrics
            node_id = int(metrics.get("node_id", -1))
            t_loss  = metrics.get("train_loss", 0.0)
            t_acc   = metrics.get("train_accuracy", 0.0)
            n_samp  = fit_res.num_examples

            train_losses.append(t_loss)
            train_accs.append(t_acc)

            logger.info(
                f"  Client {node_id}: loss={t_loss:.4f} | "
                f"acc={t_acc:.4f} | samples={n_samp}"
            )

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_train_acc  = np.mean(train_accs)   if train_accs   else 0.0

        logger.info(f"  Agregat: loss={avg_train_loss:.4f} | acc={avg_train_acc:.4f}")

        # Simpan ke history
        self.history["rounds"].append(server_round)
        self.history["train_loss"].append(float(avg_train_loss))
        self.history["train_accuracy"].append(float(avg_train_acc))

        # Panggil agregasi FedAvg standar
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        return aggregated_params, aggregated_metrics

    # ── aggregate_evaluate ─────────────────────────
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Agregasi hasil evaluasi dari semua klien."""

        if not results:
            return None, {}

        # Weighted average loss
        total_examples = sum(r.num_examples for _, r in results)
        weighted_loss  = sum(
            r.loss * r.num_examples for _, r in results
        ) / total_examples

        # Weighted average accuracy
        weighted_acc = sum(
            r.metrics.get("accuracy", 0.0) * r.num_examples
            for _, r in results
        ) / total_examples

        logger.info(f"\n  ── Evaluasi Round {server_round} ──")
        logger.info(f"  Eval loss    : {weighted_loss:.4f}")
        logger.info(f"  Eval accuracy: {weighted_acc:.4f}")

        # Simpan ke history
        if len(self.history["eval_loss"]) < server_round:
            self.history["eval_loss"].append(float(weighted_loss))
            self.history["eval_accuracy"].append(float(weighted_acc))

        # Simpan model terbaik
        if weighted_acc > self.best_accuracy:
            self.best_accuracy = weighted_acc
            logger.info(f"  ★ Model terbaik baru! Akurasi={weighted_acc:.4f}")
            # Parameter terbaik disimpan dari callback di server
            self._save_history()

        return weighted_loss, {
            "accuracy": weighted_acc,
            "loss": weighted_loss,
        }

    # ── Utility ────────────────────────────────────
    def _save_history(self):
        """Simpan history training ke file JSON."""
        history_path = os.path.join(self.save_dir, "fl_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.debug(f"History disimpan ke {history_path}")

    def print_summary(self):
        """Cetak ringkasan hasil FL setelah semua round selesai."""
        logger.info("\n" + "═" * 60)
        logger.info("  RINGKASAN FL TRAINING — FedSentiment")
        logger.info("═" * 60)

        if self.history["eval_accuracy"]:
            best_round = int(np.argmax(self.history["eval_accuracy"])) + 1
            best_acc   = max(self.history["eval_accuracy"])
            last_acc   = self.history["eval_accuracy"][-1]
            logger.info(f"  Round terbaik   : {best_round}")
            logger.info(f"  Akurasi terbaik : {best_acc:.4f} ({best_acc*100:.2f}%)")
            logger.info(f"  Akurasi akhir   : {last_acc:.4f} ({last_acc*100:.2f}%)")

        if self.history["train_loss"]:
            logger.info(f"  Train loss akhir: {self.history['train_loss'][-1]:.4f}")

        logger.info("═" * 60)
        self._save_history()
