# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

Multimodal emotion recognition course project (TCU, Third Year, Second Semester, Deep Learning). A **text + face image** dual-model emotion classification system built from scratch with PyTorch, deployed via Streamlit.

## Project architecture

Two independent models — no fusion, no shared weights:

| Model | Architecture | Params | Val Acc | Checkpoint |
|-------|-------------|--------|---------|------------|
| Image | 4-layer CNN (ImageEmotionClassifier) | 5.6M | 87%+ | `models/best_image_model.pth` |
| Text | BiLSTM (TextEmotionClassifier) | 4M | 94.81% | `models/best_text_model.pth` |

**Inference path** (`src/inference.py` → `EmotionPredictor`):
- Loads both models independently
- Image: Haar Cascade face detection → 48×48 grayscale → CNN → 7-class softmax
- Text: tokenize → BiLSTM → 7-class softmax
- No audio, no cross-modal fusion at inference time

**Legacy code** (kept for reference, not used by the app):
- `src/model.py` — MultiModalEmotionNet (3-encoder fusion with audio)
- `src/trainer.py` — Fusion model trainer
- `src/train.py` — Fusion model training entry point

## Running the app

```bash
# Install deps
pip install -r requirements.txt

# Launch Streamlit
streamlit run app.py
```

The app supports three input modes: webcam capture, file upload, and text input.

## Training

```bash
python src/image_trainer.py    # Train standalone image CNN
python src/text_trainer.py     # Train standalone text BiLSTM
```

Both trainers auto-create `data/`, `models/`, `results/` directories and save checkpoints. Each uses the `Trainer` class pattern: model, optimizer, criterion, data loaders, and training loop are owned by the trainer.

## Key source files

| File | Role |
|------|------|
| `src/config.py` | All hyperparameters, paths, device selection |
| `src/image_model.py` | ImageEmotionClassifier — 4-conv CNN |
| `src/text_model.py` | TextEmotionClassifier — Embedding + BiLSTM |
| `src/image_trainer.py` | Image model training with AMP, class-balanced sampling |
| `src/text_trainer.py` | Text model training with AMP |
| `src/inference.py` | EmotionPredictor — dual-model inference (used by app.py) |
| `src/preprocess.py` | TextPreprocessor, ImagePreprocessor, AudioPreprocessor |
| `src/dataset.py` | MultiModalEmotionDataset, create_dataloaders |
| `src/model.py` | MultiModalEmotionNet — legacy fusion model |
| `src/trainer.py` | EmotionTrainer — legacy fusion trainer |
| `src/train.py` | Legacy fusion training entry |
| `src/build_final_dataset.py` | Dataset construction from raw data |
| `src/build_texts.py` | Text corpus builder |
| `app.py` | Streamlit UI — webcam, upload, text input |

## Data files

- `data/text_vocab.pkl` — 5K word vocabulary (required for inference)
- `data/*_samples.json` — train/val/test split indices
- `data/all_images.npy` — FER2013 face images as numpy array (316 MB, gitignored)
- `data/text_corpus.json`, `data/goemotions_texts.json`, `data/real_texts.json` — text training corpora (gitignored)

## Dependencies

Core for inference: `torch`, `numpy`, `pillow`, `opencv-python-headless`, `streamlit`
Training extras: `matplotlib`, `scikit-learn`, `tqdm`
Data preprocessing: `librosa` (audio MFCC, legacy)

`torchvision` is NOT used — removed from requirements.

## Git notes

- Models are tracked in git (`models/*.pth` under 100MB each)
- Large data files (`.npy`, `.json`) are gitignored
- `.venv/`, `__pycache__/`, `.idea/`, `.superpowers/` are gitignored
- Streamlit Cloud deploys from this repo; models must be in git for deployment

## Streamlit Cloud

Repo: `https://github.com/Royiii3/multimodal-emotion-recognition.git`
Main file: `app.py`
Models are in-repo (no external storage needed).
