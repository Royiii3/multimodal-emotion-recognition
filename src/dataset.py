"""
多模态情感数据集 Dataset 类
优化版: 图像从预处理的 .npy 内存数组直接读取，无磁盘I/O、无MTCNN
"""

import os
import sys
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.preprocess import TextPreprocessor, ImagePreprocessor, AudioPreprocessor, EMOTION_TO_IDX, EMOTION_MAP


class MultiModalEmotionDataset(Dataset):
    """多模态情感识别数据集"""

    def __init__(
        self,
        split: str = "train",          # "train" | "val" | "test"
        data_dir: str = DATA_DIR,
        augment: bool = False,         # 是否启用数据增强
        modality_dropout: bool = False, # 是否随机丢弃模态
    ):
        self.split = split
        self.data_dir = data_dir
        self.augment = augment and (split == "train")
        self.modality_dropout = modality_dropout and (split == "train")

        # 加载样本索引
        samples_path = os.path.join(data_dir, f"{split}_samples.json")
        if not os.path.exists(samples_path):
            raise FileNotFoundError(
                f"样本文件 {samples_path} 不存在！\n"
                f"请先运行 preprocess.py 预处理数据集。"
            )

        with open(samples_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        print(f"[{split.upper()}] Loaded {len(self.samples)} samples")

        # 加载词表
        vocab_path = os.path.join(data_dir, "text_vocab.pkl")
        if os.path.exists(vocab_path):
            self.text_processor = TextPreprocessor()
            self.text_processor.load(vocab_path)
        else:
            self.text_processor = None
            print("  [WARN] No vocab found, text will be encoded on-the-fly")

        # 加载预处理的图像数组（内存映射，零I/O）
        images_path = os.path.join(data_dir, "all_images.npy")
        if os.path.exists(images_path):
            self.all_images = np.load(images_path, mmap_mode='r')
            self._use_preloaded_images = True
            print(f"  [FAST] Preloaded images: {self.all_images.shape}")
        else:
            self.all_images = None
            self._use_preloaded_images = False

        # Always create image_processor as fallback for disk-loaded images
        self.image_processor = ImagePreprocessor()
        self.audio_processor = AudioPreprocessor()

        # 情感标签映射
        self.label2idx = EMOTION_TO_IDX

    def __len__(self):
        return len(self.samples)

    def _process_text(self, sample: dict) -> torch.Tensor:
        """处理文本模态"""
        text = sample.get("text", "")
        if not text or (self.modality_dropout and random.random() < MODALITY_DROP_PROB):
            return torch.zeros(MAX_TEXT_LEN, dtype=torch.long)

        if self.text_processor:
            return self.text_processor.encode(text)
        else:
            # 临时处理
            tokens = text.lower().strip().split()
            ids = [hash(t) % 5000 + 4 for t in tokens[:MAX_TEXT_LEN]]
            ids += [0] * (MAX_TEXT_LEN - len(ids))
            return torch.tensor(ids, dtype=torch.long)

    def _process_image(self, sample: dict) -> torch.Tensor:
        """处理图像模态 — 从预处理的numpy数组直接读取，零I/O"""
        # 随机丢弃
        if self.modality_dropout and random.random() < MODALITY_DROP_PROB:
            return torch.zeros(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

        # 优先从预加载数组读取（极速！）
        if self._use_preloaded_images and "image_idx" in sample:
            idx = sample["image_idx"]
            img = self.all_images[idx]  # [48, 48] 已归一化到[-1,1]
            # 数据增强：随机水平翻转
            if self.augment and random.random() < IMAGE_AUG_PROB:
                img = np.fliplr(img).copy()
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, 48, 48]

        # 降级：从路径加载（带人脸检测）
        if "image_path" in sample and sample["image_path"]:
            path = sample["image_path"]
            if not os.path.exists(path):
                return torch.zeros(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return torch.zeros(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            return self.image_processor.process(img, detect_face=False)  # FER2013已裁剪

        return torch.zeros(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    def _process_audio(self, sample: dict) -> torch.Tensor:
        """处理语音模态"""
        if self.modality_dropout and random.random() < MODALITY_DROP_PROB:
            return torch.zeros(N_MFCC, MAX_AUDIO_FRAMES)

        audio_path = sample.get("audio_path", "")

        # 检查是否是预处理的.pt文件
        if audio_path and audio_path.endswith('.pt'):
            if os.path.exists(audio_path):
                mfcc = torch.load(audio_path, weights_only=True)
                return mfcc
            else:
                return torch.zeros(N_MFCC, MAX_AUDIO_FRAMES)

        # 检查是否是音频文件
        if audio_path and os.path.exists(audio_path):
            return self.audio_processor.load_and_process(audio_path, augment=self.augment)

        # 无音频
        return torch.zeros(N_MFCC, MAX_AUDIO_FRAMES)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # 三模态特征
        text_tensor = self._process_text(sample)
        image_tensor = self._process_image(sample)
        audio_tensor = self._process_audio(sample)

        # 标签
        label_name = sample.get("label", "neutral")
        label_name = EMOTION_MAP.get(label_name, label_name)
        if label_name not in EMOTION_LABELS:
            label_name = "neutral"
        label_idx = self.label2idx.get(label_name, 0)

        return {
            "text": text_tensor,        # [max_len]
            "image": image_tensor,      # [1, 48, 48]
            "audio": audio_tensor,      # [N_MFCC, T]
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_name": label_name,
        }


def create_dataloaders(batch_size: int = BATCH_SIZE):
    """创建训练/验证/测试 DataLoader"""
    train_dataset = MultiModalEmotionDataset(
        split="train", augment=True, modality_dropout=True
    )
    val_dataset = MultiModalEmotionDataset(
        split="val", augment=False, modality_dropout=False
    )
    test_dataset = MultiModalEmotionDataset(
        split="test", augment=False, modality_dropout=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"DataLoaders: Train={len(train_loader)} batches, "
          f"Val={len(val_loader)} batches, "
          f"Test={len(test_loader)} batches")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试Dataset
    ds = MultiModalEmotionDataset(split="train", augment=True, modality_dropout=True)
    print(f"\nDataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
