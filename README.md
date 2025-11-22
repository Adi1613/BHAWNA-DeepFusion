Multimodal Emotion Recognition using EEG, Audio, and Video (BHAWNA Dataset)
This repository contains the full implementation of a multimodal emotion-recognition pipeline built on top of the BHAWNA dataset, an in-house naturalistic multimodal emotion dataset consisting of EEG signals, audio, and face video recordings.
The system integrates EEGNet, FaceNet, and Wave2Vec2 representation learning pipelines and performs end-to-end multimodal fusion using PyTorch and Metaflow.

🌟 Key Features
Multimodal Inputs


EEG signals → EEGNet-style encoder

Audio spectrograms → WaveNet / Wav2Vec2-style encoder

Video sequences → FaceNet-inspired 3D CNN encoder

End-to-end Training with Metaflow


📁 Project Structure
.
├── flows/
│   └── emotion_flow.py        # Metaflow pipeline orchestrating the experiment
│
├── models/
│   ├── backbone.py            # EEGNet, FaceNet, WaveNet/Wav2Vec2 style encoders
│   └── model.py               # Fusion model + classifier
│
├── training/
│   └── train_eval.py          # Training and evaluation loops
│
├── utils/
│   ├── dataset.py             # Multimodal dataset loader
│   ├── audio_utils.py         # Mel-spectrogram extraction
│   └── video_utils.py         # Frame extraction
│
├── config.py                  # Hyperparameters + training configuration
│
├── outputs/
│   ├── features/              # Cached EEG / Audio / Video embeddings
│   └── plots/                 # Loss curves, accuracy curves
│
├── README.md                  # <--- This file


📦 Installation
1. Clone repository
git clone https://github.com/<your-username>/<repo>.git
cd <repo>

2. Create environment
conda create -n emotion python=3.12
conda activate emotion

3. Install dependencies
pip install -r requirements.txt


🧪 Running the Pipeline
The entire experiment is orchestrated by Metaflow using EmotionFlow.
Default run
python flows/emotion_flow.py run

Run with specific parameters
python flows/emotion_flow.py run \
    --use_eegnet True \
    --use_facenet True \
    --use_wavenet True \
    --regen_features False \
    --device mps


🧠 Model Details
EEG Encoder (EEGNet-inspired)
Depthwise and separable 1D convolutions

Compact architecture optimized for EEG signals

Video Encoder (FaceNet-style)
3D → 2D CNN

Learns compact face embeddings

Inspired by triplet-learning FaceNet

Audio Encoder (WaveNet/Wav2Vec2-style)
Mel spectrogram extraction

2D CNN backbone

Placeholder for full WaveNet/Wav2Vec2 integration

Fusion Strategy
Concatenate latent features:
 [
 z = [z_{video} , || , z_{audio} , || , z_{eeg}]
 ]


Fully connected layers + dropout

Softmax emotion classifier

Trains using cross-entropy loss



🔧 Configuration
The config.py file defines all tunable parameters:
LEARNING_RATE
BATCH_SIZE
EPOCHS
DROPOUT_RATE
LR_STEP_SIZE, LR_GAMMA
DEVICE (cpu / cuda / mps)
Feature dimensions for each backbone
Paths for data and cache



📊 Results
Emotion Classification Performance
Multiclass classification across 7 emotions


Higher class accuracy for Happiness and Neutral, consistent with dataset priors


Confusion matrix and training curves provided in outputs/plots/


Training Visualizations
Training loss vs epoch


Validation accuracy vs epoch


Participant-wise emotion distribution plots



📄 BHAWNA Dataset
This project is based on the BHAWNA (BeHavior of Affect Within Natural Articulation) dataset:
15 participants

Hindi conversations


Multimodal signals: EEG, Audio, Video
Emotion annotation every 5 seconds
Naturalistic, context-dependent emotional content
The dataset is currently not public and is used for academic experimentation.

🧭 Future Work
Integrate full FaceNet pretrained embedding extraction
Add wav2vec 2.0 pretrained model for raw audio
Support late fusion, attention fusion, cross-modal transformers
Hyperparameter tuning via Hyperopt or Optuna
Deployment with TorchScript / ONNX


📚 Citation
If you use this repository, please cite:
Aditya Kumar Shah, Dr Virender Kadyan UPES.
Multimodal Emotion Recognition using EEG, Audio, and Video on the BHAWNA Dataset.


🙌 Acknowledgments
Special thanks to:
Machine Intelligence Research Center, UPES
BHAWNA dataset contributors
Open-source models: EEGNet, FaceNet, Wav2Vec2
Metaflow community
