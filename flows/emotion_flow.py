# flows/emotion_flow.py

import os
import glob
import random
import time

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metaflow import FlowSpec, step, Parameter, current

import sys

# Make project root importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import DatasetCfg, ModalityCfg, TrainCfg
from utils.features import generate_and_save_features
from utils.dataset import NPZDataset
from utils.collate import make_collate
from models.model import MMModel
from training.train_eval import train_epoch, eval_epoch_metrics

MC = ModalityCfg()


def _pick_device(req: str) -> str:
    req = (req or "auto").lower()
    if req != "auto":
        return req
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class EmotionFlow(FlowSpec):
    # -------- CLI Parameters --------
    root = Parameter("root", help="Root data directory", default="REPLACE_ME")
    modality_policy = Parameter("modality_policy", default="eeg_video")
    epochs = Parameter("epochs", default=10)
    batch_size = Parameter("batch_size", default=2)

    plot_curves = Parameter(
        "plot_curves",
        default=False,
        type=bool,
        help="If True, save training and validation curves to ./outputs",
    )

    regen_features = Parameter(
        "regen_features",
        default=True,
        type=bool,
        help="If False, reuse .npz features from outdir/features when available.",
    )

    # Backbone selection flags
    use_eegnet = Parameter(
        "use_eegnet",
        default=False,
        type=bool,
        help="If True, use EEGNetBackbone for EEG instead of simple EEG1D.",
    )
    use_facenet = Parameter(
        "use_facenet",
        default=False,
        type=bool,
        help="If True, use FaceNetBackbone for video instead of Video2DBackbone.",
    )
    use_wavenet = Parameter(
        "use_wavenet",
        default=False,
        type=bool,
        help="If True, use WaveNetBackbone for audio instead of AudioCNN.",
    )

    device = Parameter("device", default="auto")
    seed = Parameter("seed", default=42)
    outdir = Parameter("outdir", default="outputs")

    @step
    def start(self):
        """Initialize config and RNG; list participants and split into train/val."""
        os.makedirs(self.outdir, exist_ok=True)

        random.seed(int(self.seed))
        np.random.seed(int(self.seed))
        torch.manual_seed(int(self.seed))

        # device: keep Parameter value but also compute a runtime device
        self.runtime_device = _pick_device(self.device)

        self.dcfg = DatasetCfg(root=self.root)
        parts = (
            self.dcfg.participants
            or sorted(
                [
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(self.dcfg.root, "P*"))
                    if os.path.isdir(p)
                ]
            )
        )
        random.Random(self.dcfg.split_seed).shuffle(parts)
        n_tr = int(len(parts) * self.dcfg.train_ratio)
        self.train_parts, self.val_parts = parts[:n_tr], parts[n_tr:]

        # Build TrainCfg with backbone flags
        self.train_cfg = TrainCfg(
            use_eegnet=bool(self.use_eegnet),
            use_facenet=bool(self.use_facenet),
            use_wavenet=bool(self.use_wavenet),
        )

        print(
            {
                "train_parts": self.train_parts,
                "val_parts": self.val_parts,
                "device_param": self.device,
                "runtime_device": self.runtime_device,
                "regen_features": self.regen_features,
                "use_eegnet": self.use_eegnet,
                "use_facenet": self.use_facenet,
                "use_wavenet": self.use_wavenet,
            }
        )

        self.next(self.feature_generation)

    @step
    def feature_generation(self):
        """Step 1: Generate & persist features (.npz per sample) for all participants."""
        self.features_dir = os.path.join(self.outdir, "features")
        os.makedirs(self.features_dir, exist_ok=True)

        # If not regenerating and directory has .npz, reuse
        existing = [
            os.path.join(self.features_dir, f)
            for f in os.listdir(self.features_dir)
            if f.endswith(".npz")
        ]

        if (not self.regen_features) and existing:
            self.feature_paths = sorted(existing)
            print(
                f"Reusing {len(self.feature_paths)} existing feature files "
                f"from {self.features_dir}"
            )
        else:
            self.feature_paths = generate_and_save_features(
                self.dcfg,
                out_dir=self.features_dir,
            )
            print(
                f"Generated {len(self.feature_paths)} npz files in {self.features_dir}"
            )

        self.next(self.create_training_set)

    @step
    def create_training_set(self):
        """Step 2: Build train/val index lists from feature files."""
        train_set: List[str] = []
        val_set: List[str] = []

        for p in self.feature_paths:
            base = os.path.basename(p)
            pid = base.split("_")[0]
            if pid in self.train_parts:
                train_set.append(p)
            else:
                val_set.append(p)

        self.train_paths, self.val_paths = train_set, val_set
        print({"train_files": len(train_set), "val_files": len(val_set)})

        self.next(self.model_training)

    @step
    def model_training(self):
        """Step 3: Train the model on precomputed features."""
        # Infer EEG channels from one sample that has eeg
        n_eeg_ch = 16
        for p in self.train_paths:
            rec = np.load(p, allow_pickle=True)
            if "eeg" in rec and rec["eeg"] is not None:
                eeg_arr = rec["eeg"]
                if eeg_arr is not None:
                    n_eeg_ch = int(eeg_arr.shape[0])
                    break

        self.n_classes = len(self.dcfg.emotion_onehot_cols)

        use_video = self.modality_policy in ("all", "eeg_video", "video_only")
        use_audio = self.modality_policy in ("all",)
        use_eeg = self.modality_policy in ("all", "eeg_video", "eeg_only")

        self.model = MMModel(
            n_eeg_ch=n_eeg_ch,
            n_classes=self.n_classes,
            use_video=use_video,
            use_audio=use_audio,
            use_eeg=use_eeg,
            dropout_p=self.train_cfg.dropout,
            train_cfg=self.train_cfg,
        ).to(self.runtime_device)

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

        collate = make_collate(self.modality_policy)

        train_loader = DataLoader(
            NPZDataset(self.train_paths),
            batch_size=int(self.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            NPZDataset(self.val_paths),
            batch_size=int(self.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )

        self.history = {"loss": [], "val_acc": []}

        for ep in range(1, int(self.epochs) + 1):
            loss = train_epoch(
                self.model,
                train_loader,
                opt,
                self.runtime_device,
                lam_va=self.train_cfg.lam_va,
            )
            metrics = eval_epoch_metrics(
                self.model,
                val_loader,
                self.runtime_device,
                n_classes=self.n_classes,
            )

            self.history["loss"].append(loss)
            self.history["val_acc"].append(metrics.get("accuracy", 0.0))

            print(
                f"Ep{ep:02d} loss={loss:.4f} "
                f"acc={metrics.get('accuracy', 0):.3f} "
                f"f1m={metrics.get('f1_macro', 0):.3f}"
            )

        # Optional: save curves
        if self.plot_curves:
            import matplotlib.pyplot as plt

            os.makedirs(self.outdir, exist_ok=True)
            epochs = range(1, len(self.history["loss"]) + 1)

            # Train loss curve
            plt.figure()
            plt.plot(epochs, self.history["loss"], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Train Loss")
            plt.title("Training Loss")
            loss_path = os.path.join(self.outdir, "train_loss_curve.png")
            plt.tight_layout()
            plt.savefig(loss_path, bbox_inches="tight", dpi=160)
            plt.close()

            # Val accuracy curve
            plt.figure()
            plt.plot(epochs, self.history["val_acc"], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Val Accuracy")
            plt.title("Validation Accuracy")
            acc_path = os.path.join(self.outdir, "val_acc_curve.png")
            plt.tight_layout()
            plt.savefig(acc_path, bbox_inches="tight", dpi=160)
            plt.close()

            print({"loss_curve": loss_path, "val_acc_curve": acc_path})

        os.makedirs("checkpoints", exist_ok=True)
        self.ckpt_path = os.path.join("checkpoints", "mm_emotion_metaflow.pt")
        torch.save(
            {"state_dict": self.model.state_dict(), "n_classes": self.n_classes},
            self.ckpt_path,
        )

        self.next(self.model_evaluation)

    @step
    def model_evaluation(self):
        """Step 4: Save confusion matrix + classification report with original labels."""
        from sklearn.metrics import classification_report
        from matplotlib import pyplot as plt

        collate = make_collate(self.modality_policy)
        loader = DataLoader(
            NPZDataset(self.val_paths),
            batch_size=int(self.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )

        metrics = eval_epoch_metrics(
            self.model,
            loader,
            self.runtime_device,
            n_classes=self.n_classes,
        )
        cm = np.array(metrics["confusion_matrix"])
        class_names = list(self.dcfg.emotion_onehot_cols)

        # Save PNG confusion
        plt.figure(
            figsize=(
                1.8 + 0.35 * len(class_names),
                1.6 + 0.35 * len(class_names),
            )
        )
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion (val)")
        os.makedirs(self.outdir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cm_png = os.path.join(self.outdir, f"{ts}_cm.png")
        plt.tight_layout()
        plt.savefig(cm_png, bbox_inches="tight", dpi=160)
        plt.close()

        # CSVs
        cm_csv = os.path.join(self.outdir, f"{ts}_cm.csv")
        pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in class_names],
            columns=[f"pred_{c}" for c in class_names],
        ).to_csv(cm_csv)

        rep = classification_report(
            metrics["_y_true"],
            metrics["_y_pred"],
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        rep_csv = os.path.join(self.outdir, f"{ts}_classification_report.csv")
        pd.DataFrame(rep).transpose().to_csv(rep_csv)

        print({"cm_png": cm_png, "cm_csv": cm_csv, "report_csv": rep_csv})
        self.next(self.end)

    @step
    def end(self):
        print("Flow complete.")


if __name__ == "__main__":
    EmotionFlow()
