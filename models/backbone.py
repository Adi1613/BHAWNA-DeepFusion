# models/backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Video2DBackbone(nn.Module):
    """
    MPS-friendly video backbone:
      - Only uses Conv2d / MaxPool2d
      - Treats video as B*T frames, then averages over time.
      Input:  (B, T, C, H, W)
      Output: (B, out_dim)
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.cnn2d = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)       # stack time into batch
        z = self.cnn2d(x)                # (B*T, 64, 1, 1)
        z = z.view(B, T, 64)             # (B, T, 64)
        z = z.mean(dim=1)                # temporal average -> (B, 64)
        z = self.fc(z)                   # (B, out_dim)
        return z



class AudioCNN(nn.Module):
    """
    Simple 2D CNN for log-mel spectrograms:
      Input:  (B, M, F)  -> we treat as (B, 1, M, F)
      Output: (B, out_dim)
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, F) -> (B, 1, M, F)
        x = x.unsqueeze(1)
        z = self.cnn(x)           # (B, 32, 1, 1)
        z = z.flatten(1)          # (B, 32)
        z = self.fc(z)            # (B, out_dim)
        return z


class EEG1D(nn.Module):
    """
    Simple 1D CNN for EEG:
      Input:  (B, Ch, T)
      Output: (B, out_dim)
    """

    def __init__(self, ch: int, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Ch, T)
        z = self.conv(x)          # (B, 64, 1)
        z = z.flatten(1)          # (B, 64)
        z = self.fc(z)            # (B, out_dim)
        return z


# ------------------------------------------------------------------
# "Pretrained-style" wrapper backbones
# (for now they reuse the simple backbones; later you can plug in real weights)
# ------------------------------------------------------------------

class FaceNetBackbone(Video2DBackbone):
    """
    Placeholder FaceNet-style backbone.

    Currently just reuses Video2DBackbone architecture, but separated as a class
    so later you can:
      - load pretrained face weights
      - change the architecture without touching the rest of the code.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__(out_dim=out_dim)


class WaveNetBackbone(AudioCNN):
    """
    Placeholder WaveNet-style audio backbone.

    Currently reuses AudioCNN, but you can later replace it with a genuine
    WaveNet / Wav2Vec2 model and keep the same interface.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__(out_dim=out_dim)


class EEGNetBackbone(EEG1D):
    """
    Placeholder EEGNet-style backbone.

    Currently reuses EEG1D, but you can later implement an EEGNet architecture
    and keep this interface unchanged.
    """

    def __init__(self, n_ch: int, out_dim: int = 128):
        super().__init__(ch=n_ch, out_dim=out_dim)
