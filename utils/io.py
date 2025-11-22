# utils/io.py

import os, glob, numpy as np, pandas as pd
from typing import Optional, Tuple
import cv2
import librosa
import numbers

try:
    import torchaudio
    _HAVE_TORCHAUDIO = True
except Exception:
    torchaudio = None
    _HAVE_TORCHAUDIO = False

try:
    import ffmpeg
    _HAVE_FFMPEG = True
except Exception:
    ffmpeg = None
    _HAVE_FFMPEG = False


def _col_case_insensitive(df: pd.DataFrame, key: str) -> str:
    """
    Return the actual column name in df that matches `key` ignoring case/whitespace.
    Raises KeyError with a helpful message if not found.
    """
    l2o = {str(c).strip().lower(): c for c in df.columns}
    lk = key.strip().lower()
    if lk not in l2o:
        raise KeyError(f"Column '{key}' not found. Available: {list(df.columns)}")
    return l2o[lk]


def read_annotations_onehot(
    path: str,
    ts_col: str,
    emo_cols,
    val_col: str,
    aro_col: str,
    win_s: float
) -> pd.DataFrame:
    """
    Read annotation file and produce one row per window with:
      t0, t1, emotion_id, valence, arousal, emotion_str

    - ts_col: timestamp column name (case-insensitive; if missing, we synthesize time)
    - emo_cols: iterable of emotion column names (case-insensitive)
    - val_col, aro_col: valence/arousal columns (case-insensitive)
    """
    df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)

    cols_lower = [str(c).strip().lower() for c in df.columns]
    # Time window [t0, t1)
    if ts_col.strip().lower() in cols_lower:
        ts = pd.to_numeric(
            df[_col_case_insensitive(df, ts_col)],
            errors="coerce"
        ).fillna(0.0).to_numpy()
        t1 = ts
        t0 = np.maximum(0.0, t1 - win_s)
    else:
        # No explicit timestamp column -> assume uniform windows
        n = len(df)
        t0 = np.arange(0, n, dtype=float) * win_s
        t1 = t0 + win_s

    # Emotions -> argmax over emo_cols
    emo_stack = [
        pd.to_numeric(
            df[_col_case_insensitive(df, c)],
            errors="coerce"
        ).fillna(0.0).to_numpy()
        for c in emo_cols
    ]
    emo_mat = np.stack(emo_stack, axis=1)
    emo_idx = emo_mat.argmax(axis=1)
    # if all zero, force class 0
    emo_idx[(emo_mat.sum(axis=1) <= 0)] = 0

    # Valence / Arousal
    val = pd.to_numeric(
        df[_col_case_insensitive(df, val_col)],
        errors="coerce"
    ).to_numpy()
    aro = pd.to_numeric(
        df[_col_case_insensitive(df, aro_col)],
        errors="coerce"
    ).to_numpy()

    out = pd.DataFrame({
        "t0": t0,
        "t1": t1,
        "emotion_id": emo_idx,
        "valence": val,
        "arousal": aro
    })
    # Keep the original emotion column names as labels (lowercased in config)
    out["emotion_str"] = [emo_cols[i] for i in emo_idx]
    return out


def _normalize_frame_np(
    frame_chw: np.ndarray,
    mean=(0.45, 0.45, 0.45),
    std=(0.225, 0.225, 0.225)
) -> np.ndarray:
    """
    Normalize a CHW image using channel-wise mean/std.
    """
    m = np.asarray(mean, dtype=np.float32)[:, None, None]
    s = np.asarray(std, dtype=np.float32)[:, None, None]
    return (frame_chw - m) / (s + 1e-8)


def _normalize_size_to_dsize(size) -> Tuple[int, int]:
    """
    Normalize `size` into an OpenCV dsize=(width, height).

    Accepts:
      - int -> (size, size)
      - (H, W) -> (W, H)
    """
    if isinstance(size, (tuple, list, np.ndarray)):
        if len(size) != 2:
            raise ValueError(f"video_size must be int or length-2 tuple, got {size}")
        # assume (H, W) and convert to (W, H) for OpenCV
        h, w = int(size[0]), int(size[1])
        return (w, h)
    elif isinstance(size, numbers.Integral):
        s = int(size)
        return (s, s)
    else:
        raise TypeError(f"video_size must be int or (H,W), got type {type(size)}")


def sample_video_clip(
    video_path: str,
    fps_out: int,
    size,
    t0: float,
    t1: float
) -> Optional[np.ndarray]:
    """
    Sample a short clip [t0, t1) from the video and return
    a tensor of shape (T, C, H, W), normalized and in float32.

    - fps_out: how many frames to sample uniformly in the window
    - size: int or (H, W); resized via OpenCV with INTER_AREA
    """
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    # frame indices
    i0 = int(t0 * fps)
    i1 = max(i0 + 1, int(t1 * fps))
    if i1 <= i0:
        cap.release()
        return None

    idxs = np.linspace(i0, i1 - 1, fps_out).astype(int)
    frames = []

    dsize = _normalize_size_to_dsize(size)

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, dsize, interpolation=cv2.INTER_AREA)
        frames.append(fr)

    cap.release()

    if len(frames) < 1:
        return None

    # pad if fewer than fps_out frames were read
    while len(frames) < fps_out:
        frames.append(frames[-1])

    arr = np.stack(frames[:fps_out], axis=0).astype(np.float32)
    # (T, H, W, C) -> (T, C, H, W) and normalize to [0,1]
    arr = np.transpose(arr, (0, 3, 1, 2)) / 255.0

    # per-frame normalization
    for i in range(arr.shape[0]):
        arr[i] = _normalize_frame_np(arr[i])

    return arr


def _decode_audio_segment_ffmpeg(
    path: str,
    sr: int,
    t0: float,
    t1: float
) -> Optional[np.ndarray]:
    if not _HAVE_FFMPEG:
        return None

    dur = max(0.0, t1 - t0)
    if dur <= 0:
        return None

    try:
        out, _ = (
            ffmpeg.input(path, ss=t0, t=dur)
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        if not out:
            return None
        wav = np.frombuffer(out, np.float32)
        if wav.size == 0:
            return None
        return wav[None, :]
    except Exception:
        return None


def extract_audio_array(
    video_path: str,
    sr: int,
    t0: float,
    t1: float
) -> Optional[np.ndarray]:
    """
    Extract mono audio [t0, t1) at sample rate sr.
    Prefers torchaudio; falls back to ffmpeg if needed.
    """
    dur = max(0.0, t1 - t0)
    if dur <= 0:
        return None

    if _HAVE_TORCHAUDIO:
        try:
            wav, native_sr = torchaudio.load(video_path)  # (C, T)
            if wav.numel() == 0:
                return None
            # mono
            wav = wav.mean(dim=0, keepdim=True)  # (1, T)
            if native_sr != sr:
                wav = torchaudio.functional.resample(wav, native_sr, sr)
            s0, s1 = int(t0 * sr), int(t1 * sr)
            s1 = min(s1, wav.shape[-1])
            if s1 - s0 <= 0:
                return None
            return wav[:, s0:s1].numpy()
        except Exception:
            # fall through to ffmpeg
            pass

    return _decode_audio_segment_ffmpeg(video_path, sr, t0, t1)


def audio_to_logmel(
    wav: np.ndarray,
    sr: int,
    n_mels: int,
    win: int,
    hop: int
) -> np.ndarray:
    """
    Convert waveform (1, T) to log-mel spectrogram (M, F).
    """
    # ensure 1D
    y = wav.squeeze(0)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=hop,
        win_length=win,
        power=2.0,
    )
    return librosa.power_to_db(S + 1e-10).astype(np.float32)


def load_eeg_segment_muse(
    eeg_csv: str,
    t0: float,
    t1: float,
    out_sr: int,
    time_offset_s: float = 0.0
) -> Optional[np.ndarray]:
    """
    Load Muse EEG CSV and return channels x T array for [t0, t1),
    resampled to out_sr, using a simple interpolation.
    """
    df = pd.read_csv(eeg_csv)

    # Find a timestamp column
    ts_col = None
    for cand in ("TimeStamp", "Timestamp", "TIME", "time", "DateTime"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError(f"No TimeStamp-like column in {eeg_csv}")

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    mask = ts.notna()
    df = df.loc[mask].copy()
    ts = ts.loc[mask]

    tsec = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()

    # numeric-ify all non-timestamp columns
    for c in df.columns:
        if c == ts_col:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    num_df = df.select_dtypes(include=[np.number])
    good = ~num_df.isna().all(axis=1)
    num_df = num_df.loc[good]
    tsec = tsec[good.to_numpy()]

    if len(num_df) < 4:
        return None

    # Try to keep relevant EEG channels
    keep = [
        c for c in num_df.columns
        if c.startswith(("RAW_", "Alpha_", "Beta_", "Theta_", "Gamma_",
                        "Delta_", "Accelerometer_", "Gyro_"))
    ]
    if keep:
        num_df = num_df[keep]

    eeg = num_df.to_numpy().T  # (Ch, N)

    # Window in EEG time
    t0e, t1e = t0 + time_offset_s, t1 + time_offset_s
    mask2 = (tsec >= t0e) & (tsec < t1e)
    if mask2.sum() < 4:
        return None

    seg_t = tsec[mask2] - t0e
    seg_e = eeg[:, mask2]
    dur = t1 - t0
    tgt = int(out_sr * dur)
    if tgt <= 1:
        return None

    grid = np.linspace(0.0, dur, tgt, endpoint=False)
    out = np.zeros((seg_e.shape[0], tgt), dtype=np.float32)
    for i in range(seg_e.shape[0]):
        out[i] = np.interp(grid, seg_t, seg_e[i])

    return out
