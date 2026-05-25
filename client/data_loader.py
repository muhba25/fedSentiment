"""
client/data_loader.py
---------------------
Modul untuk memuat dan mempartisi dataset IMDB untuk Federated Learning.

Dataset: IMDB Movie Reviews
  - 25.000 data training (label: positif/negatif)
  - 25.000 data test
  - Dibagi rata ke 3 klien (partisi IID)

Tokenisasi: Sederhana berbasis word-level dengan vocab 20.000 kata.
"""

import os
import re
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Dict, Optional
from collections import Counter
from loguru import logger

# Gunakan HuggingFace datasets jika tersedia, fallback ke dummy
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("HuggingFace datasets tidak tersedia. Gunakan dataset dummy.")

# Lokasi dataset lokal relatif terhadap file ini
_LOCAL_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


# ══════════════════════════════════════════════
#  TOKENIZER SEDERHANA
# ══════════════════════════════════════════════

class SimpleTokenizer:
    """
    Tokenizer word-level sederhana.
    Tidak memerlukan dependensi berat seperti tokenizer transformer.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, vocab_size: int = 20000, max_len: int = 256):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.is_fitted = False

    @staticmethod
    def _preprocess(text: str) -> List[str]:
        """Bersihkan teks dan tokenisasi per kata."""
        text = text.lower()
        # Hapus tag HTML
        text = re.sub(r"<[^>]+>", " ", text)
        # Pertahankan huruf dan spasi saja
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def fit(self, texts: List[str]) -> "SimpleTokenizer":
        """Bangun vocabulary dari corpus."""
        counter = Counter()
        for text in texts:
            counter.update(self._preprocess(text))

        # Ambil top (vocab_size - 2) kata (PAD dan UNK sudah ada)
        most_common = counter.most_common(self.vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.is_fitted = True
        logger.info(f"Vocab size: {len(self.word2idx):,} kata")
        return self

    def encode(self, text: str) -> List[int]:
        """Encode teks → list token ID (dengan padding/truncation)."""
        tokens = self._preprocess(text)
        ids = [self.word2idx.get(w, 1) for w in tokens]  # 1 = UNK

        # Truncate atau pad ke max_len
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [0] * (self.max_len - len(ids))  # PAD = 0

        return ids

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch teks → numpy array [N, max_len]."""
        return np.array([self.encode(t) for t in texts], dtype=np.int64)


# ══════════════════════════════════════════════
#  DATASET CLASS
# ══════════════════════════════════════════════

class IMDBDataset(Dataset):
    """Dataset PyTorch untuk IMDB reviews."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer
        self.labels = torch.tensor(labels, dtype=torch.long)

        logger.info(f"Tokenisasi {len(texts)} sampel...")
        self.input_ids = torch.tensor(
            tokenizer.encode_batch(texts), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


# ══════════════════════════════════════════════
#  DATASET DUMMY (fallback)
# ══════════════════════════════════════════════

def generate_dummy_imdb(
    n_samples: int = 3000,
    vocab_size: int = 20000,
    max_len: int = 256,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Generate dummy dataset yang mensimulasikan IMDB.
    Digunakan jika HuggingFace datasets tidak tersedia atau untuk testing cepat.
    """
    rng = np.random.RandomState(seed)

    positive_words = [
        "great", "excellent", "amazing", "wonderful", "fantastic",
        "brilliant", "superb", "outstanding", "perfect", "loved",
        "enjoyed", "beautiful", "impressive", "masterpiece", "gem",
    ]
    negative_words = [
        "terrible", "awful", "horrible", "worst", "boring",
        "disappointing", "waste", "bad", "poor", "dull",
        "mediocre", "failed", "ridiculous", "unbearable", "garbage",
    ]
    neutral_words = [
        "movie", "film", "story", "character", "plot", "scene",
        "director", "actor", "performance", "acting", "watch",
        "show", "episode", "series", "cinema", "screen",
    ]

    texts, labels = [], []
    for i in range(n_samples):
        label = rng.randint(0, 2)
        n_words = rng.randint(30, 100)

        words = rng.choice(neutral_words, size=n_words // 2, replace=True).tolist()
        if label == 1:
            sentiment_words = rng.choice(positive_words, size=n_words // 2, replace=True)
        else:
            sentiment_words = rng.choice(negative_words, size=n_words // 2, replace=True)
        words += sentiment_words.tolist()
        rng.shuffle(words)

        texts.append(" ".join(words))
        labels.append(label)

    logger.info(f"Dataset dummy: {n_samples} sampel ({sum(labels)} positif, {n_samples - sum(labels)} negatif)")
    return texts, labels


# ══════════════════════════════════════════════
#  FUNGSI UTAMA
# ══════════════════════════════════════════════

def _load_csv(path: str) -> Tuple[List[str], List[int]]:
    """Baca CSV dengan kolom 'text' dan 'label'."""
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def _local_dataset_available() -> bool:
    train = os.path.join(_LOCAL_DATASET_DIR, "imdb_train.csv")
    test  = os.path.join(_LOCAL_DATASET_DIR, "imdb_test.csv")
    return os.path.isfile(train) and os.path.isfile(test)


def load_imdb_data(
    use_dummy: bool = False,
    max_samples: int = None,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Muat dataset IMDB. Prioritas: dataset lokal → HuggingFace → dummy.
    Returns: (train_texts, train_labels, test_texts, test_labels)
    """
    if use_dummy:
        logger.info("Menggunakan dataset DUMMY...")
        train_texts, train_labels = generate_dummy_imdb(n_samples=3000, seed=42)
        test_texts, test_labels   = generate_dummy_imdb(n_samples=600,  seed=99)
        return train_texts, train_labels, test_texts, test_labels

    if _local_dataset_available():
        logger.info(f"Memuat dataset lokal dari {_LOCAL_DATASET_DIR} ...")
        train_texts, train_labels = _load_csv(os.path.join(_LOCAL_DATASET_DIR, "imdb_train.csv"))
        test_texts,  test_labels  = _load_csv(os.path.join(_LOCAL_DATASET_DIR, "imdb_test.csv"))
        logger.info(f"Dataset lokal — train: {len(train_texts):,} | test: {len(test_texts):,}")
    elif HF_AVAILABLE:
        logger.info("Mengunduh dataset IMDB dari HuggingFace...")
        dataset = load_dataset("imdb", trust_remote_code=True)
        train_texts  = dataset["train"]["text"]
        train_labels = dataset["train"]["label"]
        test_texts   = dataset["test"]["text"]
        test_labels  = dataset["test"]["label"]
        logger.info(f"IMDB train: {len(train_texts):,} | test: {len(test_texts):,}")
    else:
        logger.warning("Dataset lokal tidak ditemukan & HuggingFace tidak tersedia. Pakai DUMMY.")
        train_texts, train_labels = generate_dummy_imdb(n_samples=3000, seed=42)
        test_texts, test_labels   = generate_dummy_imdb(n_samples=600,  seed=99)
        return train_texts, train_labels, test_texts, test_labels

    if max_samples:
        train_texts  = train_texts[:max_samples]
        train_labels = train_labels[:max_samples]
        test_texts   = test_texts[:max_samples // 5]
        test_labels  = test_labels[:max_samples // 5]

    return train_texts, train_labels, test_texts, test_labels


def partition_data(
    texts: List[str],
    labels: List[int],
    num_clients: int = 3,
    node_id: int = 0,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Partisi dataset secara IID ke beberapa klien.
    Setiap klien mendapat shard yang sama besar.
    
    Args:
        texts       : Semua teks
        labels      : Semua label
        num_clients : Total jumlah klien FL
        node_id     : ID klien ini (0-indexed)
        seed        : Random seed untuk reproducibility
    
    Returns:
        (texts_shard, labels_shard) untuk klien ini
    """
    assert 0 <= node_id < num_clients, f"node_id harus antara 0 dan {num_clients - 1}"

    # Shuffle dengan seed yang sama di semua node agar partisi konsisten
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(texts))

    # Bagi indices ke num_clients shard
    shards = np.array_split(indices, num_clients)
    client_indices = shards[node_id]

    client_texts  = [texts[i] for i in client_indices]
    client_labels = [labels[i] for i in client_indices]

    n_pos = sum(client_labels)
    n_neg = len(client_labels) - n_pos
    logger.info(
        f"[Node {node_id}] Shard: {len(client_labels)} sampel "
        f"(+{n_pos} positif, -{n_neg} negatif)"
    )
    return client_texts, client_labels


def get_data_loaders(
    node_id: int,
    config: dict,
    use_dummy: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Fungsi utama: kembalikan (train_loader, test_loader) untuk node tertentu.

    Args:
        node_id   : ID klien (0, 1, atau 2)
        config    : Konfigurasi dari config.yaml
        use_dummy : Gunakan dataset dummy jika True

    Returns:
        (train_loader, test_loader)
    """
    vocab_size  = config["model"]["vocab_size"]
    max_len     = config["training"]["max_seq_len"]
    batch_size  = config["training"]["batch_size"]
    num_clients = config["dataset"]["num_clients"]
    seed        = config["dataset"]["seed"]

    # 1. Muat data
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(
        use_dummy=use_dummy,
        max_samples=config["dataset"].get("max_samples"),
    )

    # 2. Partisi data training ke klien ini
    client_texts, client_labels = partition_data(
        train_texts, train_labels,
        num_clients=num_clients,
        node_id=node_id,
        seed=seed,
    )

    # 3. Bangun tokenizer dari data KLIEN INI saja (simulasi FL: data tidak berbagi)
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_len=max_len)
    tokenizer.fit(client_texts)

    # 4. Buat dataset
    train_dataset = IMDBDataset(client_texts, client_labels, tokenizer)

    # Partisi test juga ke klien ini untuk evaluasi lokal
    test_texts_c, test_labels_c = partition_data(
        test_texts, test_labels,
        num_clients=num_clients,
        node_id=node_id,
        seed=seed + 1,
    )
    test_dataset = IMDBDataset(test_texts_c, test_labels_c, tokenizer)

    # 5. Buat DataLoader
    # num_workers=0 wajib saat berjalan di dalam Flower client subprocess.
    # Flower sudah menggunakan multiprocessing; DataLoader workers tambahan
    # (num_workers > 0) akan fork di dalam fork → crash "worker exited unexpectedly".
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    logger.info(
        f"[Node {node_id}] Train: {len(train_dataset)} | "
        f"Test: {len(test_dataset)} | "
        f"Batches: {len(train_loader)} train, {len(test_loader)} test"
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Test loader
    import yaml
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)

    for node in range(3):
        train_loader, test_loader = get_data_loaders(node_id=node, config=config, use_dummy=True)
        batch = next(iter(train_loader))
        print(f"Node {node}: input={batch[0].shape}, label={batch[1].shape}")
    print("✓ Data loader OK")
