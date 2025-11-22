import torch

def make_collate(policy: str):
    def pad_time_video(x: torch.Tensor, T: int) -> torch.Tensor:
        t = x.shape[0]
        if t == T: return x
        if t > T:  return x[:T]
        pad = x[-1:].expand(T - t, *x.shape[1:])
        return torch.cat([x, pad], dim=0)

    def pad_last_dim(x: torch.Tensor, L: int) -> torch.Tensor:
        cur = x.shape[-1]
        if cur == L: return x
        if cur > L:  return x[..., :L]
        pad = x[..., -1:].expand(*list(x.shape[:-1]), L - cur)
        return torch.cat([x, pad], dim=-1)

    def collate(batch):
        def has(b, k): return b[k] is not None
        if policy == "all":
            kept = [b for b in batch if has(b,'video') and has(b,'audio_mel') and has(b,'eeg')]
        elif policy == "eeg_video":
            kept = [b for b in batch if has(b,'video') and has(b,'eeg')]
        elif policy == "eeg_only":
            kept = [b for b in batch if has(b,'eeg')]
        elif policy == "video_only":
            kept = [b for b in batch if has(b,'video')]
        else:
            kept = []
        if not kept: return None
        vids = mels = eegs = None
        if any(has(b,'video') for b in kept):
            T = max(b['video'].shape[0] for b in kept if b['video'] is not None)
            vids = torch.stack([pad_time_video(b['video'], T) for b in kept if b['video'] is not None], dim=0)
        if any(has(b,'audio_mel') for b in kept):
            L = max(b['audio_mel'].shape[-1] for b in kept if b['audio_mel'] is not None)
            mels = torch.stack([pad_last_dim(b['audio_mel'], L) for b in kept if b['audio_mel'] is not None], dim=0)
        if any(has(b,'eeg') for b in kept):
            L = max(b['eeg'].shape[-1] for b in kept if b['eeg'] is not None)
            eegs = torch.stack([pad_last_dim(b['eeg'], L) for b in kept if b['eeg'] is not None], dim=0)
        y = torch.stack([b['y_cls'] for b in kept], dim=0)
        return vids, mels, eegs, y
    return collate

#########
def make_collate_pretrained():
    def collate(batch):
        kept = [b for b in batch if b["y_cls"] is not None]
        if not kept:
            return None

        def stack_or_none(key):
            items = [b[key] for b in kept if b[key] is not None]
            if not items:
                return None
            return torch.stack(items, dim=0)

        eeg_emb   = stack_or_none("eeg_emb")
        face_emb  = stack_or_none("face_emb")
        audio_emb = stack_or_none("audio_emb")
        y_cls = torch.stack([b["y_cls"] for b in kept], dim=0)

        return eeg_emb, face_emb, audio_emb, y_cls

    return collate
##########
