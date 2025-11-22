# models/model.py

from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn

from config import TrainCfg
from models.backbone import (
    Video2DBackbone,
    AudioCNN,
    EEG1D,
    FaceNetBackbone,
    WaveNetBackbone,
    EEGNetBackbone,
)


class FusionHead(nn.Module):
    """
    Simple fusion + 2-head output:
      - emotion classification (logits)
      - valence/arousal regression (2-dim)
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        dropout_p: float = 0.2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(hidden_dim, n_classes)
        self.fc_va = nn.Linear(hidden_dim, 2)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(self.act(self.fc1(z)))
        logits = self.fc_cls(h)
        va_hat = self.fc_va(h)
        return logits, va_hat


class MMModel(nn.Module):
    """
    Multimodal model with pluggable backbones.

    Inputs:
      - video: (B, T, C, H, W) or None
      - audio_mel: (B, M, F) or None
      - eeg: (B, Ch, T_eeg) or None

    Backbones selected via TrainCfg:
      - EEG: EEG1D vs EEGNetBackbone
      - Video: Video2DBackbone vs FaceNetBackbone
      - Audio: AudioCNN vs WaveNetBackbone
    """

    def __init__(
        self,
        n_eeg_ch: int,
        n_classes: int,
        use_video: bool = True,
        use_audio: bool = True,
        use_eeg: bool = True,
        dropout_p: float = 0.2,
        train_cfg: Optional[TrainCfg] = None,
    ):
        super().__init__()
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_eeg = use_eeg

        cfg = train_cfg or TrainCfg()
        # handy for debugging / logging
        _ = asdict(cfg)

        vid_dim = aud_dim = eeg_dim = 0

        # --- Video branch ---
        if self.use_video:
            if cfg.use_facenet:
                self.video_enc = FaceNetBackbone(out_dim=256)
            else:
                self.video_enc = Video2DBackbone(out_dim=256)
            vid_dim = 256
        else:
            self.video_enc = None

        # --- Audio branch ---
        if self.use_audio:
            if cfg.use_wavenet:
                self.audio_enc = WaveNetBackbone(out_dim=128)
            else:
                self.audio_enc = AudioCNN(out_dim=128)
            aud_dim = 128
        else:
            self.audio_enc = None

        # --- EEG branch ---
        if self.use_eeg:
            if cfg.use_eegnet:
                self.eeg_enc = EEGNetBackbone(n_ch=n_eeg_ch, out_dim=128)
            else:
                self.eeg_enc = EEG1D(ch=n_eeg_ch, out_dim=128)
            eeg_dim = 128
        else:
            self.eeg_enc = None

        in_fuse = vid_dim + aud_dim + eeg_dim
        if in_fuse == 0:
            raise ValueError("MMModel received no active modalities (all use_* flags are False).")

        self.fuse = FusionHead(
            in_dim=in_fuse,
            n_classes=n_classes,
            dropout_p=dropout_p,
            hidden_dim=128,
        )

    def forward(
        self,
        video: Optional[torch.Tensor],
        audio_mel: Optional[torch.Tensor],
        eeg: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = []

        if self.use_video and video is not None:
            feats.append(self.video_enc(video))

        if self.use_audio and audio_mel is not None:
            feats.append(self.audio_enc(audio_mel))

        if self.use_eeg and eeg is not None:
            feats.append(self.eeg_enc(eeg))

        if not feats:
            raise ValueError("No modalities were provided to MMModel.forward (all inputs None).")

        z = torch.cat(feats, dim=1)
        logits, va_hat = self.fuse(z)
        return logits, va_hat
