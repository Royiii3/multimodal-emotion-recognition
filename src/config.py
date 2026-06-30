"""
Configuration — Text + Image + Audio multimodal emotion recognition
(Note: audio modality is deprecated for inference but kept for model/checkpoint compat)
"""

import torch
import os

# ==================== Paths ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== Device ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True if DEVICE.type == "cuda" else False

# ==================== Dataset ====================
NUM_CLASSES = 7
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==================== Text preprocessing ====================
MAX_VOCAB_SIZE = 5000
MAX_TEXT_LEN = 64
TEXT_EMBED_DIM = 256

# ==================== Image preprocessing ====================
IMAGE_SIZE = 48
IMAGE_CHANNELS = 1
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5

# ==================== Audio preprocessing (legacy — kept for compat) ====================
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 3.0
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
MAX_AUDIO_FRAMES = 300

# ==================== Audio augmentation (legacy) ====================
AUDIO_SPECAUGMENT = True
AUDIO_TIME_MASK = 10
AUDIO_FREQ_MASK = 5

# ==================== Model ====================
# Text encoder
TEXT_LSTM_HIDDEN = 256
TEXT_LSTM_LAYERS = 2
TEXT_LSTM_DROPOUT = 0.3
TEXT_LSTM_BIDIRECTIONAL = True

# Image encoder
IMAGE_CONV_CHANNELS = [32, 64, 128, 256]
IMAGE_KERNEL_SIZE = 3
IMAGE_POOL_SIZE = 2

# Audio encoder (legacy — kept for model compat)
AUDIO_CONV_KERNEL = 3
AUDIO_LSTM_HIDDEN = 128
AUDIO_LSTM_LAYERS = 1
AUDIO_LSTM_DROPOUT = 0.3

# Fusion
FUSION_ATTENTION_HEADS = 4
FUSION_HIDDEN_DIM = 512
FUSION_DROPOUT = 0.5

# Classifier
CLASSIFIER_HIDDEN = 256
CLASSIFIER_DROPOUT = 0.5

# ==================== Training ====================
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8

LR_T_0 = 10
LR_T_MULT = 2
WARMUP_EPOCHS = 2

# ==================== Augmentation ====================
IMAGE_AUG_PROB = 0.5

# ==================== Regularization ====================
MODALITY_DROP_PROB = 0.0

# ==================== DataLoader ====================
NUM_WORKERS = 0      # Windows 安全值 (mmap 不可跨进程序列化)
PIN_MEMORY = True if DEVICE.type == "cuda" else False

# ==================== Logging & save ====================
LOG_INTERVAL = 50
SAVE_BEST_MODEL = True
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pth")


def print_config():
    """Print all config for training confirmation"""
    print("=" * 60)
    print(f"{'Config':<35} {'Value':<25}")
    print("=" * 60)
    print(f"{'DEVICE':<35} {str(DEVICE):<25}")
    print(f"{'USE_AMP':<35} {str(USE_AMP):<25}")
    print(f"{'NUM_CLASSES':<35} {NUM_CLASSES:<25}")
    print(f"{'BATCH_SIZE':<35} {BATCH_SIZE:<25}")
    print(f"{'LEARNING_RATE':<35} {LEARNING_RATE:<25}")
    print(f"{'MAX_EPOCHS':<35} {MAX_EPOCHS:<25}")
    print(f"{'TEXT_EMBED_DIM':<35} {TEXT_EMBED_DIM:<25}")
    print(f"{'IMAGE_SIZE':<35} {f'{IMAGE_SIZE}x{IMAGE_SIZE}':<25}")
    print(f"{'FUSION_HIDDEN_DIM':<35} {FUSION_HIDDEN_DIM:<25}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
