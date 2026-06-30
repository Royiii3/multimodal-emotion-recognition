"""
数据预处理流水线
- Plan A: CMU-MOSEI 三模态对齐数据
- Plan B: FER2013 + GoEmotions + RAVDESS 独立数据集拼接

流程: 原始数据 → 文本分词/图像人脸检测/MFCC提取 → 保存.pt张量
"""

import os
import sys
import json
import pickle
import random
from collections import Counter

import numpy as np
import torch

# torchaudio 可能因CUDA版本问题不可用，提供librosa降级方案
try:
    import torchaudio
    import torchaudio.transforms as T
    _HAS_TORCHAUDIO = True
except (ImportError, OSError):
    _HAS_TORCHAUDIO = False
    print("[WARN] torchaudio not available, using librosa for MFCC extraction")

import librosa
import cv2
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *


# ==================== 共同情感标签映射 ====================
# 将不同数据集的情感标签统一映射到标准8类
EMOTION_MAP = {
    # FER2013
    "0": "angry", "1": "disgust", "2": "fearful", "3": "happy",
    "4": "sad", "5": "surprised", "6": "neutral",
    # GoEmotions (简化映射)
    "anger": "angry", "disgust": "disgust", "fear": "fearful",
    "joy": "happy", "sadness": "sad", "surprise": "surprised",
    "neutral": "neutral", "love": "happy", "amusement": "happy",
    "excitement": "happy",
    # RAVDESS (calm removed — not in EMOTION_LABELS)
    "01": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

EMOTION_TO_IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}


# ==================== 文本预处理 ====================
class TextPreprocessor:
    """文本分词 + 词表构建 + 编码"""

    def __init__(self, max_vocab=MAX_VOCAB_SIZE, max_len=MAX_TEXT_LEN):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.vocab_size = 4

    def tokenize(self, text: str) -> list:
        """简单英文分词: 按空格切分，去标点"""
        text = text.lower().strip()
        # 基本标点替换为空格
        for ch in ['.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '-']:
            text = text.replace(ch, ' ')
        return [t for t in text.split() if t]

    def build_vocab(self, texts: list):
        """从文本列表构建词表"""
        word_counts = Counter()
        for text in tqdm(texts, desc="Building vocab"):
            tokens = self.tokenize(text)
            word_counts.update(tokens)

        # 取 top (max_vocab - 4) 个词
        for word, _ in word_counts.most_common(self.max_vocab - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"Vocab size: {self.vocab_size}")

    def encode(self, text: str) -> torch.Tensor:
        """文本 → token id 序列"""
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(t, 1) for t in tokens]  # 未登录词→<UNK>
        # 截断/填充
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word,
                         "vocab_size": self.vocab_size, "max_len": self.max_len}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = data["idx2word"]
        self.vocab_size = data["vocab_size"]
        self.max_len = data["max_len"]


# ==================== 图像预处理 ====================
class ImagePreprocessor:
    """人脸检测 + 裁剪 + 归一化"""

    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size
        self.face_detector = None  # 延迟加载MTCNN

    def _init_face_detector(self):
        """延迟初始化MTCNN（避免未安装时直接报错）"""
        if self.face_detector is None:
            try:
                from facenet_pytorch import MTCNN
                self.face_detector = MTCNN(
                    image_size=self.image_size, margin=10,
                    keep_all=False, post_process=False, device=DEVICE
                )
            except ImportError:
                print("[WARN] facenet-pytorch 未安装，使用OpenCV Haar Cascade降级方案")
                self.face_detector = "opencv"

    def detect_face_opencv(self, img: np.ndarray) -> np.ndarray:
        """OpenCV Haar Cascade 人脸检测（降级方案）"""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # 取最大人脸
            face = gray[y:y+h, x:x+w]
            return cv2.resize(face, (self.image_size, self.image_size))
        else:
            # 无人脸 → 缩放整张图
            return cv2.resize(gray, (self.image_size, self.image_size))

    def process(self, img: np.ndarray, detect_face: bool = True) -> torch.Tensor:
        """
        输入: numpy图像 (H, W) 灰度 或 (H, W, 3) 彩色
        输出: tensor [1, 48, 48], 归一化到 [-1, 1]
        """
        if detect_face:
            self._init_face_detector()
            if isinstance(self.face_detector, str) and self.face_detector == "opencv":
                img = self.detect_face_opencv(img)
            elif self.face_detector is not None:
                # MTCNN 处理
                from PIL import Image
                if len(img.shape) == 2:
                    pil_img = Image.fromarray(img).convert('RGB')
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                face = self.face_detector(pil_img)
                if face is None:
                    # 无人脸
                    img = cv2.resize(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                    (self.image_size, self.image_size))
                else:
                    img = face.squeeze(0).numpy().mean(axis=0)  # RGB→灰度
        else:
            # 不做人脸检测，直接resize
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.image_size, self.image_size))

        # 归一化
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGE_MEAN) / IMAGE_STD
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]


# ==================== 语音预处理 ====================
class AudioPreprocessor:
    """MFCC特征提取 — 优先torchaudio，降级librosa"""

    def __init__(self):
        if _HAS_TORCHAUDIO:
            self.mfcc_transform = T.MFCC(
                sample_rate=AUDIO_SAMPLE_RATE,
                n_mfcc=N_MFCC,
                melkwargs={
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "win_length": WIN_LENGTH,
                    "n_mels": 64,
                }
            )
            self.time_mask = T.TimeMasking(time_mask_param=AUDIO_TIME_MASK)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=AUDIO_FREQ_MASK)

    def _extract_mfcc_librosa(self, waveform_np: np.ndarray, sr: int) -> np.ndarray:
        """使用librosa提取MFCC"""
        mfcc = librosa.feature.mfcc(
            y=waveform_np, sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )  # [N_MFCC, T]
        return mfcc

    def _trim_or_pad(self, array: np.ndarray, target_size: int, axis: int = -1) -> np.ndarray:
        """截断或填充数组到固定大小"""
        if array.shape[axis] > target_size:
            if axis == -1 or axis == array.ndim - 1:
                return array[..., :target_size]
            else:
                return array[:target_size]
        elif array.shape[axis] < target_size:
            pad_width = [(0, 0)] * array.ndim
            pad_width[axis] = (0, target_size - array.shape[axis])
            return np.pad(array, pad_width, mode='constant')
        return array

    def _specaugment_np(self, mfcc: np.ndarray) -> np.ndarray:
        """numpy实现的简易SpecAugment"""
        # 时间遮罩
        t = np.random.randint(0, AUDIO_TIME_MASK)
        t0 = np.random.randint(0, max(1, mfcc.shape[1] - t))
        mfcc[:, t0:t0 + t] = 0
        # 频率遮罩
        f = np.random.randint(0, AUDIO_FREQ_MASK)
        f0 = np.random.randint(0, max(1, mfcc.shape[0] - f))
        mfcc[f0:f0 + f, :] = 0
        return mfcc

    def load_and_process(self, audio_path: str, augment: bool = False) -> torch.Tensor:
        """
        加载音频文件 → 重采样 → MFCC → 截断/填充 → 增强
        输出: [N_MFCC, MAX_AUDIO_FRAMES]
        """
        if _HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != AUDIO_SAMPLE_RATE:
                resampler = T.Resample(sr, AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)
            target_len = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            else:
                pad = torch.zeros(1, target_len - waveform.shape[1])
                waveform = torch.cat([waveform, pad], dim=1)
            mfcc = self.mfcc_transform(waveform).squeeze(0)
            mean, std = mfcc.mean(), mfcc.std() + 1e-8
            mfcc = (mfcc - mean) / std
            # 截断/填充时间维度
            if mfcc.shape[1] > MAX_AUDIO_FRAMES:
                mfcc = mfcc[:, :MAX_AUDIO_FRAMES]
            elif mfcc.shape[1] < MAX_AUDIO_FRAMES:
                pad = torch.zeros(N_MFCC, MAX_AUDIO_FRAMES - mfcc.shape[1])
                mfcc = torch.cat([mfcc, pad], dim=1)
            if augment and AUDIO_SPECAUGMENT:
                mfcc = self.time_mask(mfcc)
                mfcc = self.freq_mask(mfcc)
            return mfcc
        else:
            # librosa fallback
            waveform, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
            target_len = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
            waveform = self._trim_or_pad(waveform, target_len, axis=0)
            mfcc = self._extract_mfcc_librosa(waveform, AUDIO_SAMPLE_RATE)
            mfcc = self._trim_or_pad(mfcc, MAX_AUDIO_FRAMES, axis=1)
            mean, std = mfcc.mean(), mfcc.std() + 1e-8
            mfcc = (mfcc - mean) / std
            if augment and AUDIO_SPECAUGMENT:
                mfcc = self._specaugment_np(mfcc)
            return torch.tensor(mfcc, dtype=torch.float32)

    def process_array(self, waveform: torch.Tensor, sr: int, augment: bool = False) -> torch.Tensor:
        """从内存中的波形数组处理"""
        # Convert to numpy and use librosa for consistency
        waveform_np = waveform.cpu().numpy()
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=0)
        waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
        target_len = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
        waveform_np = self._trim_or_pad(waveform_np, target_len, axis=0)
        mfcc = self._extract_mfcc_librosa(waveform_np, AUDIO_SAMPLE_RATE)
        mfcc = self._trim_or_pad(mfcc, MAX_AUDIO_FRAMES, axis=1)
        mean, std = mfcc.mean(), mfcc.std() + 1e-8
        mfcc = (mfcc - mean) / std
        if augment and AUDIO_SPECAUGMENT:
            mfcc = self._specaugment_np(mfcc)
        return torch.tensor(mfcc, dtype=torch.float32)


# ==================== FER2013 数据集加载 ====================
def load_fer2013(csv_path: str) -> tuple:
    """
    加载FER2013 CSV文件
    返回: (images: np.ndarray [N,48,48], labels: list of int)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    for _, row in df.iterrows():
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
        images.append(pixels.reshape(48, 48))
        labels.append(int(row['emotion']))
    return np.array(images), labels


# ==================== GoEmotions 数据集加载 ====================
def load_goemotions(data_dir: str, split: str = "train") -> tuple:
    """
    加载GoEmotions数据集 (HuggingFace格式)
    返回: (texts: list of str, labels: list of str)
    """
    from datasets import load_dataset
    dataset = load_dataset("go_emotions", split=split)
    texts = []
    labels = []
    for item in dataset:
        texts.append(item["text"])
        # GoEmotions 是多标签，取第一个标签作为主情感
        if len(item["labels"]) > 0:
            labels.append(item["labels"][0])
        else:
            labels.append(-1)  # neutral

    # 将数字标签映射到情感名
    emotion_names = dataset.features["labels"].feature.names
    label_names = [emotion_names[l] if l >= 0 else "neutral" for l in labels]
    return texts, label_names


# ==================== RAVDESS 数据集加载 ====================
def load_ravdess(audio_dir: str) -> tuple:
    """
    加载RAVDESS语音情感数据集
    文件名格式: 03-01-01-01-01-01-01.wav
               ↑ 第3位是情感标签: 01=neutral, 02=calm, 03=happy, ...
    返回: (audio_paths: list, emotion_labels: list of str)
    """
    audio_paths = []
    labels = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith('.wav'):
            continue
        parts = fname.split('-')
        if len(parts) < 3:
            continue
        emotion_code = parts[2]  # 第三段是情感编码
        if emotion_code in EMOTION_MAP:
            audio_paths.append(os.path.join(audio_dir, fname))
            labels.append(EMOTION_MAP[emotion_code])
    return audio_paths, labels


# ==================== 主预处理流程 ====================
def build_fer2013_goemotions_ravdess_dataset():
    """
    Plan B: 拼接三数据集 → 按情感标签对齐 → 构建多模态样本

    输出文件:
    - data/text_vocab.pkl      词表
    - data/train_samples.json  训练样本索引
    - data/val_samples.json    验证样本索引
    - data/test_samples.json   测试样本索引
    - data/preprocessed/       预处理后的张量
    """
    print("=" * 60)
    print("Plan B: FER2013 + GoEmotions + RAVDESS 数据集预处理")
    print("=" * 60)

    preprocessed_dir = os.path.join(DATA_DIR, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    # --- 1. 加载文本数据 (GoEmotions) ---
    print("\n[1/5] Loading GoEmotions...")
    try:
        train_texts, train_text_labels = load_goemotions(None, "train")
        val_texts, val_text_labels = load_goemotions(None, "validation")
        test_texts, test_text_labels = load_goemotions(None, "test")
        all_texts = train_texts + val_texts + test_texts
        all_text_labels = train_text_labels + val_text_labels + test_text_labels
        print(f"  Loaded {len(all_texts)} text samples")
    except Exception as e:
        print(f"  [WARN] GoEmotions load failed: {e}")
        print("  Using synthetic demo dataset (50 sentences)")
        all_texts, all_text_labels = _create_demo_texts()
        train_texts, val_texts, test_texts = all_texts[:30], all_texts[30:40], all_texts[40:]
        train_text_labels = all_text_labels[:30]
        val_text_labels = all_text_labels[30:40]
        test_text_labels = all_text_labels[40:]

    # 构建词表
    text_preprocessor = TextPreprocessor()
    text_preprocessor.build_vocab(all_texts)
    text_preprocessor.save(os.path.join(DATA_DIR, "text_vocab.pkl"))

    # --- 2. 加载图像数据 (FER2013) ---
    print("\n[2/5] Loading FER2013...")
    fer2013_path = os.path.join(DATA_DIR, "fer2013.csv")
    if os.path.exists(fer2013_path):
        images, img_labels = load_fer2013(fer2013_path)
        print(f"  Loaded {len(images)} face images")
        # 将数字标签映射到情感名
        img_label_names = [EMOTION_MAP[str(l)] for l in img_labels]
    else:
        print(f"  [WARN] {fer2013_path} not found!")
        print("  Please download from: https://www.kaggle.com/datasets/msambare/fer2013")
        print("  Using random demo data (500 samples)")
        images = np.random.randint(0, 255, (500, 48, 48), dtype=np.uint8)
        img_label_names = [random.choice(EMOTION_LABELS[:7]) for _ in range(500)]

    # --- 3. 加载语音数据 (RAVDESS) ---
    print("\n[3/5] Loading RAVDESS...")
    ravdess_dir = os.path.join(DATA_DIR, "ravdess")
    if os.path.exists(ravdess_dir):
        audio_paths, audio_labels = load_ravdess(ravdess_dir)
        print(f"  Loaded {len(audio_paths)} audio clips")
    else:
        print(f"  [WARN] {ravdess_dir} not found!")
        print("  Please download from Kaggle: RAVDESS Emotional Speech Audio")
        print("  Using synthetic demo MFCC data (300 samples)")
        audio_paths = []
        audio_labels = []
        for i in range(300):
            label = random.choice(EMOTION_LABELS)
            synth_path = os.path.join(preprocessed_dir, f"synth_audio_{i}.pt")
            # 生成随机MFCC
            rand_mfcc = torch.randn(N_MFCC, MAX_AUDIO_FRAMES)
            torch.save(rand_mfcc, synth_path)
            audio_paths.append(synth_path)
            audio_labels.append(label)

    # --- 4. 按情感标签对齐 + 划分 ---
    print("\n[4/5] Aligning modalities by emotion labels...")

    # 按标签组织样本
    def group_by_label(texts, labels):
        grouped = {}
        for t, l in zip(texts, labels):
            l_std = EMOTION_MAP.get(l, l) if l in EMOTION_MAP else l
            if l_std not in EMOTION_LABELS:
                l_std = "neutral"
            grouped.setdefault(l_std, []).append(t)
        return grouped

    text_grouped = group_by_label(all_texts, all_text_labels)
    img_grouped = group_by_label(images, img_label_names)
    audio_grouped = group_by_label(audio_paths, audio_labels)

    # 为每个样本创建多模态三元组
    samples = []
    for emotion in EMOTION_LABELS:
        t_pool = text_grouped.get(emotion, [])
        i_pool = img_grouped.get(emotion, [])
        a_pool = audio_grouped.get(emotion, [])

        if not t_pool or not i_pool or not a_pool:
            print(f"  [WARN] {emotion}: insufficient modality data, skipping")
            continue

        # 取各模态最小数量作为该情感样本数
        n = min(len(t_pool), len(i_pool), len(a_pool))
        # 随机配对
        random.shuffle(t_pool)
        random.shuffle(i_pool)
        random.shuffle(a_pool)

        for j in range(n):
            samples.append({
                "text": t_pool[j],
                "image_idx": j if isinstance(i_pool[j], np.ndarray) else j,
                "image_is_path": not isinstance(i_pool[j], np.ndarray),
                "image": i_pool[j] if isinstance(i_pool[j], np.ndarray) else None,
                "image_path": i_pool[j] if not isinstance(i_pool[j], np.ndarray) else None,
                "audio_path": a_pool[j],
                "label": emotion,
            })

    print(f"  Total multimodal samples: {len(samples)}")

    # --- 5. 划分 + 保存 ---
    print("\n[5/5] Splitting and saving...")
    random.seed(42)
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split_name, split_samples in splits.items():
        print(f"  {split_name}: {len(split_samples)} samples")
        with open(os.path.join(DATA_DIR, f"{split_name}_samples.json"), 'w', encoding='utf-8') as f:
            json.dump(split_samples, f, ensure_ascii=False, indent=2)

    print("\n[OK] Preprocessing complete! Output files:")
    print(f"   {DATA_DIR}/text_vocab.pkl")
    for s in ["train", "val", "test"]:
        print(f"   {DATA_DIR}/{s}_samples.json")
    print(f"   {DATA_DIR}/preprocessed/ (audio .pt files)")

    return splits


def _create_demo_texts():
    """创建演示文本数据集（当GoEmotions不可用时使用）"""
    demos = [
        ("I am so happy today everything is wonderful", "joy"),
        ("This makes me really angry and frustrated", "anger"),
        ("I feel so sad and lonely right now", "sadness"),
        ("I am terrified of what might happen next", "fear"),
        ("That is absolutely disgusting", "disgust"),
        ("What a wonderful surprise this is amazing", "surprise"),
        ("I dont have any particular feelings about this", "neutral"),
        ("The weather is nice and I am feeling great", "joy"),
        ("I hate when people are so cruel", "anger"),
        ("Tears fell down my face as I remembered", "sadness"),
        ("The dark shadows made me nervous", "fear"),
        ("The rotten smell was revolting", "disgust"),
        ("I never expected this incredible gift", "surprise"),
        ("It is just another ordinary day", "neutral"),
        ("My heart is full of joy and gratitude", "joy"),
        ("I could barely control my fury", "anger"),
        ("A deep sorrow filled my heart", "sadness"),
        ("I was shaking with fright", "fear"),
        ("The sight was utterly repulsive", "disgust"),
        ("Such an unexpected turn of events", "surprise"),
        ("I feel calm and content today", "neutral"),
        ("This is the best day of my life", "joy"),
        ("Dont you dare talk to me like that", "anger"),
        ("I miss you so much it hurts", "sadness"),
        ("Something terrible is about to happen", "fear"),
        ("That food was absolutely nasty", "disgust"),
        ("Wow I did not see that coming", "surprise"),
        ("Nothing special happened today", "neutral"),
        ("I am truly blessed and thankful", "joy"),
        ("You are making me so mad right now", "anger"),
        ("My heart aches with grief", "sadness"),
        ("I am scared of the dark", "fear"),
        ("This garbage smells awful", "disgust"),
        ("Oh my god are you serious", "surprise"),
        ("Just a normal Tuesday afternoon", "neutral"),
        ("Laughing with friends is the best feeling", "joy"),
        ("I will not tolerate this disrespect", "anger"),
        ("The loneliness is overwhelming", "sadness"),
        ("Please dont hurt me", "fear"),
        ("The moldy bread was disgusting", "disgust"),
        ("I am shocked by this revelation", "surprise"),
        ("I have no strong opinion on this matter", "neutral"),
        ("Pure bliss and happiness overflow", "joy"),
        ("Get out of my face right now", "anger"),
        ("Why does everything go wrong for me", "sadness"),
        ("That horror movie traumatized me", "fear"),
        ("Throw that nasty thing away", "disgust"),
        ("What an astonishing performance", "surprise"),
        ("Lets just keep things as they are", "neutral"),
    ]
    texts = [d[0] for d in demos]
    labels = [d[1] for d in demos]
    return texts, labels


if __name__ == "__main__":
    build_fer2013_goemotions_ravdess_dataset()
