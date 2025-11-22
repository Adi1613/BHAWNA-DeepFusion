# 🚀 Running the Multimodal Emotion Recognition Flow (Local)

### **Run via Metaflow CLI**

To execute the full multimodal emotion recognition pipeline locally:

```bash
python -m metaflow run flows/emotion_flow.py \
    --root /path/to/data \
    --modality_policy eeg_video \
    --epochs 10 \
    --batch_size 2
```

### **Run directly (without `-m metaflow`)**

```bash
python flows/emotion_flow.py run \
    --root /Users/adityashah/Documents/PhD/PhD/data \
    --modality_policy eeg_video \
    --epochs 10 \
    --batch_size 2 \
    --plot_curves True \
    --regen_features False
```

You can also run it inside a **Jupyter Notebook / Colab**:

```python
!python flows/emotion_flow.py run --root /path/to/data --modality_policy eeg_video
```

or Metaflow magic:

```python
%run flows/emotion_flow.py --root /path/to/data --modality_policy eeg_video
```

---

# 📁 Expected Data Layout

Your dataset root directory must follow this structure:

```
ROOT/
├── P1/
│   ├── P1_annotation.csv   (or .xlsx)
│   ├── P1_eeg.csv
│   └── *.mp4               (video files)
│
├── P2/
│   ├── P2_annotation.csv
│   ├── P2_eeg.csv
│   └── *.mp4
│
└── ...
```

**Notes:**

* Annotation file must contain timestamp + emotion labels.
* Each MP4 contains continuous video for that participant.
* EEG is raw Muse (27-channel) time-series data.

---

# 📦 Outputs Generated

After running the flow, you will find:

### **1. Feature Cache**

```
outputs/features/*.npz
```

Preprocessed 5-second aligned multimodal windows (video embeddings, log-mel audio, EEG features).

### **2. Model Checkpoints**

```
checkpoints/mm_emotion_metaflow.pt
```

Saved multimodal fusion transformer weights.

### **3. Evaluation Artifacts**

```
outputs/train_loss_curve.png
outputs/val_acc_curve.png
outputs/confusion_matrix.png
outputs/*.csv
```

Includes:

* Confusion matrix (PNG + CSV)
* Classification report
* Per-class metrics (precision/recall/F1)
* Training/validation learning curves


