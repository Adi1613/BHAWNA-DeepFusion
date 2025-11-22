from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class DatasetCfg:
    root: str
    participants: Optional[List[str]] = None
    emotion_onehot_cols: Tuple[str, ...] = (
        "neutral",
        "anger",
        "disgust",
        "fear",
        "happiness",  # ✅ must match the Excel column
        "sadness",    # ✅ must match the Excel column
        "surprise",
    )
    ann_timestamp_col: str = "Timestamp"
    ann_valence_col: str = "Valence"
    ann_arousal_col: str = "Arousal"
    ann_window_seconds: float = 5.0
    eeg_video_time_offset: Optional[Dict[str, float]] = None
    split_seed: int = 42
    train_ratio: float = 0.8


@dataclass
class ModalityCfg:
    """
    Simple container for modality-specific options.
    """
    video_fps: int = 25
    video_num_frames: int = 16

    audio_sample_rate: int = 16000
    audio_window_sec: float = 4.0

    eeg_sample_rate: int = 256
    eeg_window_sec: float = 4.0
    n_eeg_channels: int = 16


# config.py (only the TrainCfg part needs to be updated)
from dataclasses import dataclass

@dataclass
class TrainCfg:
    lr: float = 2e-4
    weight_decay: float = 1e-2
    lam_va: float = 0.5          # weight for valence/arousal regression loss
    dropout: float = 0.2

    # NEW: backbone selection flags
    use_eegnet: bool = False     # if True -> EEGNetBackbone, else simple EEG1D
    use_facenet: bool = False    # if True -> FaceNetBackbone, else Video2DBackbone
    use_wavenet: bool = False    # if True -> WaveNetBackbone, else AudioCNN
