"""
utils/metrics.py
----------------
Utilitas untuk menghitung, menampilkan, dan memvisualisasikan
metrik hasil Federated Learning.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (server environment)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional
from loguru import logger


# ══════════════════════════════════════════════
#  KLASIFIKASI METRIK
# ══════════════════════════════════════════════

def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Hitung akurasi klasifikasi."""
    if not y_true:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int = 2,
) -> np.ndarray:
    """Hitung confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def compute_f1(y_true: List[int], y_pred: List[int], pos_label: int = 1) -> float:
    """Hitung F1-score untuk kelas positif."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ══════════════════════════════════════════════
#  VISUALISASI FL TRAINING
# ══════════════════════════════════════════════

def plot_fl_history(
    history: Dict,
    save_path: str = "results/fl_training_curve.png",
    title: str = "FedSentiment — FL Training History",
) -> None:
    """
    Plot kurva training dan evaluasi selama FL rounds.
    
    Args:
        history  : Dict dengan keys: rounds, train_loss, train_accuracy,
                   eval_loss, eval_accuracy
        save_path: Path untuk menyimpan gambar
        title    : Judul plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rounds       = history.get("rounds", [])
    train_loss   = history.get("train_loss", [])
    train_acc    = history.get("train_accuracy", [])
    eval_loss    = history.get("eval_loss", [])
    eval_acc     = history.get("eval_accuracy", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # ── Plot 1: Loss ──────────────────────────
    ax1 = axes[0]
    if train_loss:
        ax1.plot(rounds[:len(train_loss)], train_loss,
                 "o-", color="#E85D24", linewidth=2, markersize=5,
                 label="Train Loss (avg klien)")
    if eval_loss:
        ax1.plot(rounds[:len(eval_loss)], eval_loss,
                 "s--", color="#1D9E75", linewidth=2, markersize=5,
                 label="Eval Loss (avg klien)")

    ax1.set_xlabel("FL Round", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Loss per FL Round", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Plot 2: Akurasi ───────────────────────
    ax2 = axes[1]
    if train_acc:
        ax2.plot(rounds[:len(train_acc)], [a * 100 for a in train_acc],
                 "o-", color="#534AB7", linewidth=2, markersize=5,
                 label="Train Accuracy (%)")
    if eval_acc:
        ax2.plot(rounds[:len(eval_acc)], [a * 100 for a in eval_acc],
                 "s--", color="#3B8BD4", linewidth=2, markersize=5,
                 label="Eval Accuracy (%)")

    # Tandai akurasi terbaik
    if eval_acc:
        best_idx = int(np.argmax(eval_acc))
        best_rnd = rounds[best_idx] if best_idx < len(rounds) else best_idx + 1
        best_val = eval_acc[best_idx] * 100
        ax2.annotate(
            f"Best: {best_val:.1f}%",
            xy=(best_rnd, best_val),
            xytext=(best_rnd + 0.5, best_val + 2),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="#3B8BD4",
        )

    ax2.set_xlabel("FL Round", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy per FL Round", fontsize=12)
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot FL history disimpan ke: {save_path}")


def print_fl_report(history: Dict) -> None:
    """Cetak laporan ringkasan FL ke console."""
    train_acc = history.get("train_accuracy", [])
    eval_acc  = history.get("eval_accuracy",  [])
    train_loss = history.get("train_loss", [])

    print("\n" + "═" * 55)
    print("  LAPORAN HASIL FEDERATED LEARNING — FedSentiment")
    print("═" * 55)

    if eval_acc:
        best_idx = int(np.argmax(eval_acc))
        print(f"  Round terbaik       : {best_idx + 1}")
        print(f"  Akurasi terbaik     : {eval_acc[best_idx] * 100:.2f}%")
        print(f"  Akurasi akhir (eval): {eval_acc[-1] * 100:.2f}%")

    if train_acc:
        print(f"  Akurasi akhir (train): {train_acc[-1] * 100:.2f}%")

    if train_loss:
        print(f"  Loss akhir (train)  : {train_loss[-1]:.4f}")

    # Tabel per-round
    print("\n  Tabel Metrik per Round:")
    print(f"  {'Round':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Eval Acc':>8}")
    print("  " + "-" * 45)

    rounds = history.get("rounds", list(range(1, max(
        len(train_acc), len(eval_acc), len(train_loss)
    ) + 1)))

    for i, r in enumerate(rounds):
        tl = f"{train_loss[i]:.4f}" if i < len(train_loss) else "  —  "
        ta = f"{train_acc[i]*100:.2f}%" if i < len(train_acc) else "  —  "
        ea = f"{eval_acc[i]*100:.2f}%"  if i < len(eval_acc)  else "  —  "
        print(f"  {r:>5} | {tl:>10} | {ta:>9} | {ea:>8}")

    print("═" * 55)


# ══════════════════════════════════════════════
#  MAIN (untuk analisis post-training)
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    history_file = sys.argv[1] if len(sys.argv) > 1 else "results/fl_history.json"

    if not os.path.exists(history_file):
        # Buat data dummy untuk demonstrasi
        logger.info("File history tidak ditemukan. Menggunakan data demo...")
        history = {
            "rounds": list(range(1, 11)),
            "train_loss": [0.693, 0.612, 0.543, 0.489, 0.441,
                           0.401, 0.370, 0.345, 0.326, 0.310],
            "train_accuracy": [0.521, 0.631, 0.702, 0.749, 0.781,
                               0.808, 0.827, 0.842, 0.853, 0.862],
            "eval_loss": [0.671, 0.598, 0.533, 0.481, 0.437,
                          0.400, 0.372, 0.349, 0.331, 0.317],
            "eval_accuracy": [0.548, 0.659, 0.726, 0.768, 0.796,
                               0.820, 0.836, 0.849, 0.858, 0.865],
        }
    else:
        with open(history_file) as f:
            history = json.load(f)

    # Cetak laporan
    print_fl_report(history)

    # Buat plot
    plot_fl_history(history, save_path="results/fl_training_curve.png")
    print(f"\nPlot disimpan ke: results/fl_training_curve.png")
