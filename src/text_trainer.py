"""
Train text-only emotion classifier on 20K real text corpus
"""
import sys, os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.text_model import TextEmotionClassifier
from src.preprocess import TextPreprocessor


def load_text_data():
    """Load text corpus, split into train/val/test"""
    corpus_path = os.path.join(DATA_DIR, 'text_corpus.json')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # Group by emotion for stratified split
    by_emotion = {}
    for e in entries:
        by_emotion.setdefault(e['label'], []).append(e['text'])

    np.random.seed(42)
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    test_texts, test_labels = [], []

    for emotion, texts in by_emotion.items():
        np.random.shuffle(texts)
        n = len(texts)
        t_end = int(n * TRAIN_RATIO)
        v_end = t_end + int(n * VAL_RATIO)
        for t in texts[:t_end]:
            train_texts.append(t); train_labels.append(emotion)
        for t in texts[t_end:v_end]:
            val_texts.append(t); val_labels.append(emotion)
        for t in texts[v_end:]:
            test_texts.append(t); test_labels.append(emotion)

    print(f'Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}')

    # Build vocab from training data only
    tp = TextPreprocessor()
    tp.build_vocab(train_texts)
    tp.save(os.path.join(DATA_DIR, 'text_vocab.pkl'))

    label2idx = {l: i for i, l in enumerate(EMOTION_LABELS)}

    # Encode all texts
    def encode(texts, labels):
        X = torch.stack([tp.encode(t) for t in texts])
        y = torch.tensor([label2idx[l] for l in labels], dtype=torch.long)
        return X, y

    X_train, y_train = encode(train_texts, train_labels)
    X_val, y_val = encode(val_texts, val_labels)
    X_test, y_test = encode(test_texts, test_labels)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), tp


def train():
    print('=' * 60)
    print('Text Emotion Classifier Training')
    print(f'Device: {DEVICE}  |  AMP: {USE_AMP}')
    print('=' * 60)

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), tp = load_text_data()

    # Create simple DataLoader (data is small enough)
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    test_ds = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    # Model
    model = TextEmotionClassifier(vocab_size=tp.vocab_size).to(DEVICE)
    print(f'Params: {model.count_parameters()["total"]:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if USE_AMP else None

    best_val_acc = 0.0
    patience = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(30):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            if USE_AMP:
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(-1) == y).sum().item()
            train_total += y.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                val_correct += (logits.argmax(-1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss/len(val_loader))
        history['val_acc'].append(val_acc)

        print(f'  Train: loss={history["train_loss"][-1]:.4f} acc={train_acc:.2%}  |  '
              f'Val: loss={history["val_loss"][-1]:.4f} acc={val_acc:.2%}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': tp.vocab_size,
                'best_val_acc': best_val_acc,
                'history': history,
                'emotion_labels': EMOTION_LABELS,
            }, os.path.join(MODEL_DIR, 'best_text_model.pth'))
            print(f'  [BEST] Saved (val_acc={val_acc:.2%})')
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    # Final eval
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(-1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    test_acc = test_correct / test_total
    print(f'\n{"="*60}')
    print(f'Test Accuracy: {test_acc:.2%}')
    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, zero_division=0))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss'])+1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[1].axhline(y=best_val_acc, color='g', linestyle='--', label=f'Best: {best_val_acc:.2%}')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'text_training_curves.png'), dpi=150)
    plt.close()
    print(f'Plots saved.')


if __name__ == '__main__':
    train()
