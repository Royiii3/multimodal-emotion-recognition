# Multimodal Emotion Recognition

Text + Face dual-model emotion recognition system.

## Models

| Model | Architecture | Accuracy | Data |
|-------|-------------|----------|------|
| Text | BiLSTM (4M params) | 94.81% | Kaggle 20K emotion texts |
| Face Image | 4-layer CNN (5.6M params) | 87%+ | FER2013 35K face images |

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py          # Train image model
python src/text_trainer.py   # Train text model
streamlit run app.py         # Launch web app
```

## Project Structure

```
├── app.py                   # Streamlit UI
├── requirements.txt
├── src/
│   ├── config.py            # Hyperparameters
│   ├── preprocess.py        # Data preprocessing
│   ├── dataset.py           # Dataset class
│   ├── model.py             # Image CNN model
│   ├── text_model.py        # Text BiLSTM model
│   ├── trainer.py           # Image model trainer
│   ├── text_trainer.py      # Text model trainer
│   ├── train.py             # Training entry
│   └── inference.py         # Dual-model inference
├── data/                    # Datasets (gitignored)
├── models/                  # Checkpoints (gitignored)
└── results/                 # Plots & reports (gitignored)
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as main file
5. Upload model files via Streamlit Secrets or rebuild on first run
