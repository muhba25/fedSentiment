"""
model/cnn_text.py
-----------------
Definisi model CNN untuk klasifikasi sentimen teks.
Menggunakan arsitektur TextCNN (Kim, 2014) dengan multiple kernel sizes.

Arsitektur:
  Embedding → [Conv1D k=3] ─┐
              [Conv1D k=4] ──→ MaxPool → Concat → Dropout → FC → Softmax
              [Conv1D k=5] ─┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np


class TextCNN(nn.Module):
    """
    CNN untuk klasifikasi teks sentimen (positif/negatif).
    Berdasarkan arsitektur: Kim, Y. (2014). Convolutional Neural Networks
    for Sentence Classification.
    """

    def __init__(
        self,
        vocab_size: int = 20000,
        embed_dim: int = 128,
        num_filters: int = 100,
        kernel_sizes: List[int] = [3, 4, 5],
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes

        # ── Embedding Layer ──
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # ── Convolutional Layers (multi-kernel) ──
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2       # same padding
            )
            for k in kernel_sizes
        ])

        # ── Dropout & Classifier ──
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        # Inisialisasi bobot
        self._init_weights()

    def _init_weights(self):
        """Inisialisasi embedding dengan distribusi uniform."""
        nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Zero-out padding embedding
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Token IDs [batch_size, seq_len]
        Returns:
            logits: [batch_size, num_classes]
        """
        # [B, seq_len] → [B, seq_len, embed_dim]
        x = self.embedding(x)

        # [B, seq_len, embed_dim] → [B, embed_dim, seq_len] (Conv1d needs channels first)
        x = x.permute(0, 2, 1)

        # Terapkan setiap conv + ReLU + global max-pool
        pooled = []
        for conv in self.convs:
            # [B, num_filters, seq_len'] → [B, num_filters, 1]
            c = F.relu(conv(x))
            c = F.adaptive_max_pool1d(c, 1)   # global max pool
            pooled.append(c.squeeze(-1))       # [B, num_filters]

        # Gabungkan semua channel: [B, num_filters * len(kernel_sizes)]
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def get_parameters(self) -> List[np.ndarray]:
        """Kembalikan parameter model sebagai list numpy array (untuk Flower)."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set parameter model dari list numpy array (untuk Flower)."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)


def get_model(config: dict) -> TextCNN:
    """Factory function untuk membuat model dari config."""
    return TextCNN(
        vocab_size=config["model"]["vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        num_filters=config["model"]["num_filters"],
        kernel_sizes=config["model"]["kernel_sizes"],
        dropout=config["model"]["dropout"],
        num_classes=config["model"]["num_classes"],
    )


def count_parameters(model: TextCNN) -> int:
    """Hitung total parameter model yang trainable."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Unit test sederhana
    model = TextCNN()
    print(f"Model: TextCNN")
    print(f"Total parameter: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randint(0, 20000, (8, 256))  # batch=8, seq=256
    output = model(dummy_input)
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Expected: [8, 2]
    print("✓ Model OK")
