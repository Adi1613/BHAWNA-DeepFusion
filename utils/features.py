# utils/features.py

import os
import glob
from typing import List, Optional, Union

import numpy as np

from config import DatasetCfg, ModalityCfg
from utils.io import (
    read_annotations_onehot,
    sample_video_clip,
    extract_audio_array,
    audio_to_logmel,
    load_eeg_segment_muse,
)

MC = ModalityCfg()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _build_samples_for_participant(dcfg: DatasetCfg, pid: str) -> List[dict]:
    """
    Build list of annotation windows (with labels + [t0, t1]) for one participant.
    One row per 5s window (or dcfg.ann_window_seconds if defined).
    """
    pdir = os.path.join(dcfg.root, pid)

    ann = sorted(glob.glob(os.path.join(pdir, f"{pid}_annotation.*")))
    eeg = sorted(glob.glob(os.path.join(pdir, f"{pid}_eeg.csv")))
    vids = sorted(glob.glob(os.path.join(pdir, "*.mp4")))

    if not ann or not eeg:
        return []

    ann_path = ann[0]
    eeg_path = eeg[0]
    vid_path = vids[0] if vids else None

    # 🔹 Be robust to different DatasetCfg versions:
    ts_col = getattr(dcfg, "ann_timestamp_col", "timestamp")
    val_col = getattr(dcfg, "ann_valence_col", "valence")
    aro_col = getattr(dcfg, "ann_arousal_col", "arousal")
    win_s = getattr(dcfg, "ann_window_seconds", 5.0)

    ann_df = read_annotations_onehot(
        ann_path,
        ts_col=ts_col,
        emo_cols=tuple(c.lower() for c in dcfg.emotion_onehot_cols),
        val_col=val_col,
        aro_col=aro_col,
        win_s=win_s,
    )

    # Here we keep the “end-based” window (t0,t1 already set in read_annotations_onehot)
    samples: List[dict] = []
    for _, r in ann_df.iterrows():
        samples.append(
            dict(
                pid=pid,
                t0=float(r["t0"]),
                t1=float(r["t1"]),
                emotion_id=int(r["emotion_id"]),
                valence=float(r["valence"]) if not np.isnan(r["valence"]) else None,
                arousal=float(r["arousal"]) if not np.isnan(r["arousal"]) else None,
                video=vid_path,
                eeg=eeg_path,
            )
        )
    return samples


def _extract_modalities_for_window(
    s: dict,
    eeg_video_offset: Optional[Union[dict, float]] = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Given a sample dict with (pid, video, eeg, t0, t1),
    extract:
      - video: (T, C, H, W) float32 or None
      - audio_mel: (T_frames, n_mels) float32 or None
      - eeg: (Ch, T_eeg) float32 or None

    eeg_video_offset can be:
      - a dict {pid -> offset_seconds}
      - a scalar offset for all participants
      - or None (no offset)
    """
    pid = s["pid"]
    t0, t1 = s["t0"], s["t1"]

    # ---- Video ----
    video_arr = None
    if s["video"] is not None:
        video_arr = sample_video_clip(
            s["video"],
            fps_out=getattr(MC, "video_fps", 25),
            size=getattr(MC, "video_size", (112, 112)),
            t0=t0,
            t1=t1,
        )
        if video_arr is not None:
            video_arr = video_arr.astype(np.float32)

    # ---- Audio -> log-mel ----
    audio_mel = None
    if s["video"] is not None:
        wav = extract_audio_array(
            s["video"],
            sr=getattr(MC, "audio_sr", getattr(MC, "audio_sample_rate", 16000)),
            t0=t0,
            t1=t1,
        )
        if wav is not None and wav.shape[-1] > 800:
            audio_mel = audio_to_logmel(
                wav,
                sr=getattr(MC, "audio_sr", getattr(MC, "audio_sample_rate", 16000)),
                n_mels=getattr(MC, "n_mels", 64),
                win=getattr(MC, "audio_win", 400),
                hop=getattr(MC, "audio_hop", 160),
            ).astype(np.float32)

    # ---- EEG ----
    eeg_arr = None

    # Resolve offset:
    time_offset = 0.0
    if isinstance(eeg_video_offset, dict):
        time_offset = float(eeg_video_offset.get(pid, 0.0))
    elif isinstance(eeg_video_offset, (int, float)):
        time_offset = float(eeg_video_offset)
    # else: leave as 0.0

    if s["eeg"] is not None:
        eeg_arr = load_eeg_segment_muse(
            s["eeg"],
            t0=t0,
            t1=t1,
            out_sr=getattr(MC, "eeg_sr", getattr(MC, "eeg_sample_rate", 256)),
            time_offset_s=time_offset,
        )
        if eeg_arr is not None:
            eeg_arr = eeg_arr.astype(np.float32)

    return video_arr, audio_mel, eeg_arr


def generate_and_save_features(dcfg: DatasetCfg, out_dir: str) -> List[str]:
    """
    Main entrypoint used by the Metaflow flow.

    For each (participant, window) we:
      - read annotations (emotion_id, valence, arousal, [t0,t1])
      - extract raw video/audio/eeg arrays
      - save to a .npz with keys:
          'video'     -> (T, C, H, W) or None
          'audio_mel' -> (T_frames, n_mels) or None
          'eeg'       -> (Ch, T_eeg) or None
          'y_cls'     -> int (emotion_id)
          'y_va'      -> (2,) float [valence, arousal]
    """
    _ensure_dir(out_dir)

    participants = dcfg.participants or sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(dcfg.root, "P*"))
        if os.path.isdir(p)
    )

    # Handle either old dcfg.eeg_video_time_offset (dict)
    # or new dcfg.eeg_video_offset_sec (scalar)
    eeg_video_offset = getattr(
        dcfg,
        "eeg_video_time_offset",
        getattr(dcfg, "eeg_video_offset_sec", None),
    )

    all_paths: List[str] = []

    for pid in participants:
        samples = _build_samples_for_participant(dcfg, pid)
        if not samples:
            continue

        for idx, s in enumerate(samples):
            video_arr, audio_mel, eeg_arr = _extract_modalities_for_window(
                s,
                eeg_video_offset=eeg_video_offset,
            )

            # Targets
            y_cls = int(s["emotion_id"])
            val = -1.0 if s["valence"] is None else float(s["valence"])
            aro = -1.0 if s["arousal"] is None else float(s["arousal"])
            y_va = np.array([val, aro], dtype=np.float32)

            # Filename: Pxx_<index>.npz
            out_path = os.path.join(out_dir, f"{pid}_{idx:05d}.npz")

            np.savez_compressed(
                out_path,
                video=video_arr,
                audio_mel=audio_mel,
                eeg=eeg_arr,
                y_cls=np.int64(y_cls),
                y_va=y_va,
            )
            all_paths.append(out_path)

    return all_paths
