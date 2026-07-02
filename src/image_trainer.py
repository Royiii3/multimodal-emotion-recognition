"""
Image-only emotion classifier training — GPU-optimized for FER2013.

Usage:
    python src/image_trainer.py

Features:
    - Reads FER2013 from kagglehub cache (48×48 grayscale, organized by emotion folder)
    - Class weights to handle imbalance (happy 16× more than disgust)
    - Heavy data augmentation (RandomHorizontalFlip, RandomRotation, RandomAffine)
    - AMP mixed precision for 2× speed on RTX
    - Large batch + gradient accumulation for stable training
    - CosineAnnealingWarmRestarts + Linear Warmup
    - Early stopping by val accuracy
"""

import sys, os, json, time, random
import numpy as np
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.image_model import ImageEmotionClassifier

# ==================== Constants ====================
FER2013_CACHE = os.path.join(
    os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets',
    'msambare', 'fer2013', 'versions', '1'
)

# FER2013 folder names → our standard labels
FER_TO_STD = {
    'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprise': 'surprised',
}
STD_LABEL_TO_IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}

# Training hyperparameters (image-only)
IMG_BATCH_SIZE = 512         # Larger batch for less noisy gradients
IMG_GRAD_ACCUM = 1
IMG_LR = 1e-2                # SGD needs higher LR than Adam
IMG_MIN_LR = 1e-6
IMG_MAX_EPOCHS = 150         # More epochs without augmentation
IMG_PATIENCE = 40
IMG_LR_T0 = 30
IMG_LR_T_MULT = 2
IMG_WARMUP = 0               # No warmup — start aggressive
IMG_WEIGHT_DECAY = 1e-4
IMG_GRAD_CLIP = 1.0

# Data augmentation
AUG_ROTATION = 15            # ±15 degrees
AUG_TRANSLATE = 0.1          # ±10% translation
AUG_SCALE = 0.1              # ±10% scale


# ==================== Dataset ====================
class FER2013Dataset(Dataset):
    """Load FER2013 images from kagglehub cache directory structure."""

    def __init__(self, split: str = 'train', augment: bool = False):
        self.split = split
        self.augment = augment
        self.samples = []  # list of (filepath, label_idx)

        cache_split = os.path.join(FER2013_CACHE, split)
        if not os.path.exists(cache_split):
            raise FileNotFoundError(
                f"FER2013 {split} not found at {cache_split}\n"
                f"Please download: kagglehub download msambare/fer2013"
            )

        for folder_name in sorted(os.listdir(cache_split)):
            folder_path = os.path.join(cache_split, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name not in FER_TO_STD:
                continue

            std_label = FER_TO_STD[folder_name]
            label_idx = STD_LABEL_TO_IDX[std_label]

            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(folder_path, fname), label_idx))

        # Shuffle for better random split behavior
        random.seed(42)
        random.shuffle(self.samples)

        # Statistics
        label_counts = Counter(l for _, l in self.samples)
        print(f"[FER2013/{split.upper()}] {len(self.samples)} images | "
              + " | ".join(f"{EMOTION_LABELS[l]}={label_counts.get(l,0)}" for l in range(NUM_CLASSES)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # Load image as grayscale
        try:
            img = Image.open(filepath).convert('L')
        except Exception:
            # Corrupted image fallback: black image
            img = Image.new('L', (48, 48), 0)

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Random rotation
            angle = random.uniform(-AUG_ROTATION, AUG_ROTATION)
            if angle != 0:
                img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            # Random affine (translation + scale)
            dx = random.uniform(-AUG_TRANSLATE, AUG_TRANSLATE) * 48
            dy = random.uniform(-AUG_TRANSLATE, AUG_TRANSLATE) * 48
            scale = 1.0 + random.uniform(-AUG_SCALE, AUG_SCALE)
            if dx != 0 or dy != 0 or scale != 1.0:
                img = img.transform(
                    (48, 48), Image.AFFINE,
                    (scale, 0, dx, 0, scale, dy),
                    resample=Image.BILINEAR, fillcolor=0,
                )

        # Convert to tensor & normalize to [-1, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - IMAGE_MEAN) / IMAGE_STD
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # [1, 48, 48]

        return img_tensor, label


# ==================== Training ====================
def compute_class_weights(dataset: FER2013Dataset) -> torch.Tensor:
    """Compute balanced class weights from dataset."""
    labels = [l for _, l in dataset.samples]
    counts = Counter(labels)
    n = len(labels)
    weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        if counts.get(i, 0) > 0:
            weights[i] = n / (NUM_CLASSES * counts[i])
        else:
            weights[i] = 1.0
    return weights


def create_dataloaders():
    """Create train/val/test dataloaders with class-balanced oversampling."""
    # Train set from FER2013 'train' folder — NO augmentation for clean signal
    train_full = FER2013Dataset(split='train', augment=False)

    # ---- Balance training set by oversampling minority classes ----
    # Group samples by label
    by_class = {l: [] for l in range(NUM_CLASSES)}
    for path, label in train_full.samples:
        by_class[label].append(path)

    TARGET_PER_CLASS = 5000  # Balanced 35K training set
    balanced_samples = []
    for label, paths in by_class.items():
        if len(paths) >= TARGET_PER_CLASS:
            chosen = random.sample(paths, TARGET_PER_CLASS)
        else:
            # Duplicate with repetition to reach target
            chosen = [paths[i % len(paths)] for i in range(TARGET_PER_CLASS)]
        balanced_samples.extend([(p, label) for p in chosen])

    random.seed(42)
    random.shuffle(balanced_samples)

    # Replace train_full samples with balanced version
    train_full.samples = balanced_samples
    counts = Counter(l for _, l in balanced_samples)
    print(f"Balanced train: {len(balanced_samples)} samples | "
          + " | ".join(f"{EMOTION_LABELS[l]}={counts[l]}" for l in range(NUM_CLASSES)))

    # Split 10% of balanced train for validation
    n_val = int(len(balanced_samples) * 0.1)
    n_train = len(balanced_samples) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Test set from FER2013 'test' folder (keep natural distribution)
    test_ds = FER2013Dataset(split='test', augment=False)

    print(f"Split: Train={n_train}, Val={n_val}, Test={len(test_ds)}")

    # No class weights needed with balanced data
    class_weights = torch.ones(NUM_CLASSES)

    # Standard shuffle
    train_loader = DataLoader(
        train_ds, batch_size=IMG_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=IMG_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds, batch_size=IMG_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, class_weights


def train():
    print("=" * 60)
    print("Image Emotion Classifier — GPU Training")
    print(f"Device: {DEVICE}  |  Batch: {IMG_BATCH_SIZE}")
    print(f"LR: {IMG_LR} → {IMG_MIN_LR}  |  Epochs: {IMG_MAX_EPOCHS}")
    print("=" * 60)

    # --- Data ---
    train_loader, val_loader, test_loader, class_weights = create_dataloaders()

    # --- Model ---
    model = ImageEmotionClassifier().to(DEVICE)
    params = model.count_parameters()
    print(f"\nModel: {params['total']:,} params ({params['trainable']:,} trainable)")

    # --- Loss: standard CE (data is already balanced via oversampling) ---
    criterion = nn.CrossEntropyLoss()
    print(f"Using standard CrossEntropyLoss (data is balanced)")

    # --- Optimizer: SGD+Momentum (better at escaping saddle points) ---
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=IMG_LR,
        momentum=0.9,
        weight_decay=IMG_WEIGHT_DECAY,
        nesterov=True,
    )

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=IMG_LR_T0, T_mult=IMG_LR_T_MULT, eta_min=IMG_MIN_LR,
    )

    # --- AMP ---
    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # --- State ---
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    # --- Warmup ---
    if IMG_WARMUP > 0:
        warmup_lr = IMG_LR * 0.1
        for pg in optimizer.param_groups:
            pg['lr'] = warmup_lr

    # ==================== Training Loop ====================
    for epoch in range(IMG_MAX_EPOCHS):
        start_time = time.time()

        # ----- Train -----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{IMG_MAX_EPOCHS} [Train]")
        for step, (images, labels) in enumerate(pbar):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if use_amp:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels) / IMG_GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                logits = model(images)
                loss = criterion(logits, labels) / IMG_GRAD_ACCUM
                loss.backward()

            # Gradient accumulation step
            if (step + 1) % IMG_GRAD_ACCUM == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), IMG_GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), IMG_GRAD_CLIP)
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * IMG_GRAD_ACCUM
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{train_loss/(step+1):.4f}',
                'acc': f'{train_correct/train_total:.3f}',
            })

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        # ----- Validate -----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{IMG_MAX_EPOCHS} [Val]  "):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                if use_amp:
                    with autocast():
                        logits = model(images)
                        loss = criterion(logits, labels)
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)

        # ----- LR Step -----
        if epoch >= IMG_WARMUP:
            scheduler.step()
        else:
            progress = (epoch + 1) / IMG_WARMUP
            lr = warmup_lr + (IMG_LR - warmup_lr) * progress
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time

        # ----- Record -----
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # ----- Print -----
        print(f"\nEpoch {epoch+1}/{IMG_MAX_EPOCHS} | Time: {elapsed:.1f}s | LR: {current_lr:.2e}")
        print(f"  Train — Loss: {train_loss_avg:.4f} | Acc: {train_acc:.2%}")
        print(f"  Val   — Loss: {val_loss_avg:.4f} | Acc: {val_acc:.2%}")

        # ----- Checkpoint -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_acc': best_val_acc,
                'class_weights': class_weights,
                'emotion_labels': EMOTION_LABELS,
                'config': {
                    'batch_size': IMG_BATCH_SIZE,
                    'lr': IMG_LR,
                    'max_epochs': IMG_MAX_EPOCHS,
                },
            }, os.path.join(MODEL_DIR, 'best_image_model.pth'))
            print(f"  ★ BEST saved (val_acc={val_acc:.2%})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  · No improvement ({patience_counter}/{IMG_PATIENCE})")

        if patience_counter >= IMG_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # ==================== Final Evaluation ====================
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")

    # Load best model for eval
    best_ckpt = torch.load(os.path.join(MODEL_DIR, 'best_image_model.pth'),
                           map_location=DEVICE, weights_only=False)
    # Actually, the model in memory IS the best one (we only save on improvement)
    # But let's reload for safety
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Test]"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = test_correct / test_total
    print(f"\nTest Accuracy: {test_acc:.2%} ({test_correct}/{test_total})")

    # Classification report
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n" + classification_report(
            all_labels, all_preds,
            target_names=EMOTION_LABELS, zero_division=0,
        ))

        # Save report
        with open(os.path.join(RESULTS_DIR, 'image_classification_report.txt'), 'w') as f:
            f.write(f"Test Accuracy: {test_acc:.2%}\n\n")
            f.write(classification_report(
                all_labels, all_preds,
                target_names=EMOTION_LABELS, zero_division=0,
            ))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels([EMOTION_CN.get(l, l) for l in EMOTION_LABELS], rotation=45, ha='right')
        ax.set_yticklabels([EMOTION_CN.get(l, l) for l in EMOTION_LABELS])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Image Classifier — Confusion Matrix')
        plt.colorbar(im)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i, j] > 0:
                    ax.text(j, i, cm[i, j], ha='center', va='center',
                            fontsize=8, color='white' if cm[i, j] > cm.max()/2 else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'image_confusion_matrix.png'), dpi=150)
        plt.close()
        print(f"Confusion matrix saved.")
    except ImportError:
        print("Install sklearn for classification report.")

    # ==================== Training curves ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[1].axhline(y=best_val_acc, color='g', linestyle='--', label=f'Best: {best_val_acc:.2%}')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR')
    axes[2].set_yscale('log'); axes[2].set_title('LR Schedule'); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'image_training_curves.png'), dpi=150)
    plt.close()
    print(f"Training curves saved.")

    print(f"\n{'='*60}")
    print(f"Training complete! Best val_acc: {best_val_acc:.2%}")
    print(f"Model saved: {MODEL_DIR}/best_image_model.pth")
    print(f"{'='*60}")

    return model, history


if __name__ == '__main__':
    # Chinese labels for plots (imported at top would fail if not defined)
    EMOTION_CN = {
        "happy": "开心", "sad": "悲伤", "angry": "愤怒",
        "fearful": "恐惧", "disgust": "厌恶", "surprised": "惊讶", "neutral": "中性",
    }
    # Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Enable for faster training

    train()
