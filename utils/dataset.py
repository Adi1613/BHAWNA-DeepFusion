# utils/dataset.py

from __future__ import annotations
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZDataset(Dataset):
    """
    Dataset over .npz feature files.

    Each .npz may contain:
      - Raw modalities:
          'video'      -> (T, C, H, W) or None (or missing)
          'audio_mel'  -> (M, F) or None (or missing)
          'eeg'        -> (Ch, T_eeg) or None (or missing)
      - OR pretrained embeddings:
          'eeg_emb'    -> (D_eeg,)  or None
          'face_emb'   -> (D_face,) or None
          'audio_emb'  -> (D_aud,)  or None

    Labels:
      - 'y_cls' (preferred)   -> int emotion id
      - or 'emotion_id' / 'y' as fallback
      - 'y_va' (optional)     -> (2,) [valence, arousal], else [-1., -1.]

    The `use_pretrained` flag controls what keys we expose:
      - use_pretrained=False  -> return raw 'video', 'audio_mel', 'eeg'
      - use_pretrained=True   -> return 'eeg_emb', 'face_emb', 'audio_emb'
    """

    def __init__(self, paths: List[str], use_pretrained: bool = False):
        self.paths = list(paths)
        self.use_pretrained = bool(use_pretrained)

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _to_float_tensor(arr: Any) -> Optional[torch.Tensor]:
        """
        Robust conversion:
          - None       -> None
          - 0-D object with None inside -> None
          - object array of arrays      -> stack if possible
          - numeric array               -> float32 tensor
        """
        if arr is None:
            return None

        # Ensure numpy array
        arr = np.array(arr, copy=False)

        # Case: pickled None -> array( None, dtype=object )
        if arr.shape == () and arr.dtype == object:
            try:
                if arr.item() is None:
                    return None
            except Exception:
                return None

        if arr.size == 0:
            return None

        # If object array (e.g., list-of-frames), try stacking
        if arr.dtype == object:
            try:
                arr = np.stack(list(arr), axis=0)
            except Exception:
                # If stacking fails, drop this modality
                return None

        # Finally, numeric -> float tensor
        return torch.from_numpy(arr.astype(np.float32))

    @staticmethod
    def _extract_label(rec, path: str) -> int:
        if "y_cls" in rec:
            return int(rec["y_cls"])
        if "emotion_id" in rec:
            return int(rec["emotion_id"])
        if "y" in rec:
            return int(rec["y"])
        raise KeyError(f"No 'y_cls', 'emotion_id', or 'y' label found in {path}")

    @staticmethod
    def _extract_va(rec) -> np.ndarray:
        if "y_va" in rec:
            y_va = np.array(rec["y_va"], dtype=np.float32)
            if y_va.shape != (2,):
                y_va = np.reshape(y_va, (2,))
        else:
            y_va = np.array([-1.0, -1.0], dtype=np.float32)
        return y_va

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        rec = np.load(path, allow_pickle=True)

        item: Dict[str, Any] = {}

        # ---------- Labels ----------
        y_cls = self._extract_label(rec, path)
        y_va = self._extract_va(rec)

        item["y_cls"] = torch.tensor(y_cls, dtype=torch.long)
        item["y_va"] = torch.from_numpy(y_va.astype(np.float32))

        # ---------- Modalities / embeddings ----------
        if not self.use_pretrained:
            # Raw modalities
            video = rec["video"] if "video" in rec else None
            audio_mel = rec["audio_mel"] if "audio_mel" in rec else None
            eeg = rec["eeg"] if "eeg" in rec else None

            item["video"] = self._to_float_tensor(video)
            item["audio_mel"] = self._to_float_tensor(audio_mel)
            item["eeg"] = self._to_float_tensor(eeg)
        else:
            # Pretrained embeddings (may be missing/None)
            eeg_emb = rec["eeg_emb"] if "eeg_emb" in rec else None
            face_emb = rec["face_emb"] if "face_emb" in rec else None
            audio_emb = rec["audio_emb"] if "audio_emb" in rec else None

            item["eeg_emb"] = self._to_float_tensor(eeg_emb)
            item["face_emb"] = self._to_float_tensor(face_emb)
            item["audio_emb"] = self._to_float_tensor(audio_emb)

        return item
