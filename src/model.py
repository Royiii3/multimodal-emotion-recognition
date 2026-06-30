"""
MultiModalEmotionNet — 多模态情感识别模型
三编码器 + 跨模态注意力融合 + 分类头

架构:
  TextEncoder:   Embedding → BiLSTM → h_text   [B, 256]
  ImageEncoder:  4-Conv CNN → GAP → h_img       [B, 256]
  AudioEncoder:  2-Conv CNN → BiLSTM → h_audio  [B, 256]
  Fusion:        concat → MultiHeadSelfAttention → FC → h_fused [B, 512]
  Classifier:    FC → ReLU → Dropout → FC → Softmax
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 文本编码器 ====================
class TextEncoder(nn.Module):
    """Embedding + 2层BiLSTM → 文本特征向量"""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=MAX_VOCAB_SIZE,
            embedding_dim=TEXT_EMBED_DIM,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=TEXT_EMBED_DIM,
            hidden_size=TEXT_LSTM_HIDDEN,
            num_layers=TEXT_LSTM_LAYERS,
            dropout=TEXT_LSTM_DROPOUT if TEXT_LSTM_LAYERS > 1 else 0,
            bidirectional=TEXT_LSTM_BIDIRECTIONAL,
            batch_first=True,
        )
        lstm_out_dim = TEXT_LSTM_HIDDEN * 2 if TEXT_LSTM_BIDIRECTIONAL else TEXT_LSTM_HIDDEN
        self.proj = nn.Linear(lstm_out_dim, 256)  # 投影到统一维度
        self.dropout = nn.Dropout(TEXT_LSTM_DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, max_len] token ids
        返回: [B, 256]
        """
        emb = self.embedding(x)                       # [B, max_len, 256]
        emb = self.dropout(emb)
        lstm_out, (h_n, _) = self.lstm(emb)           # [B, max_len, H*2]
        # 取最后一层两个方向的最后hidden拼接
        if TEXT_LSTM_BIDIRECTIONAL:
            h_forward = h_n[-2, :, :]   # 正向最后层
            h_backward = h_n[-1, :, :]  # 反向最后层
            h = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h = h_n[-1, :, :]
        return self.proj(h)                           # [B, 256]


# ==================== 图像编码器 ====================
class ImageEncoder(nn.Module):
    """4层CNN → 面部表情特征向量"""

    def __init__(self):
        super().__init__()
        layers = []
        in_ch = IMAGE_CHANNELS
        for i, out_ch in enumerate(IMAGE_CONV_CHANNELS):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, IMAGE_KERNEL_SIZE, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(IMAGE_POOL_SIZE),
                nn.Dropout2d(0.1) if i < 2 else nn.Identity(),  # 浅层加轻微dropout
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        # AdaptiveAvgPool → 256 维
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(IMAGE_CONV_CHANNELS[-1], 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 48, 48] 灰度人脸
        返回: [B, 256]
        """
        feat = self.conv(x)              # [B, 256, 6, 6]
        feat = self.gap(feat).flatten(1) # [B, 256]
        return self.proj(feat)           # [B, 256]


# ==================== 语音编码器 ====================
class AudioEncoder(nn.Module):
    """2层CNN + 1层BiLSTM → 语音情感特征向量"""

    def __init__(self):
        super().__init__()
        # Conv1D over MFCC
        self.conv = nn.Sequential(
            nn.Conv1d(N_MFCC, 64, AUDIO_CONV_KERNEL, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, AUDIO_CONV_KERNEL, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        # BiLSTM over temporal features
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=AUDIO_LSTM_HIDDEN,
            num_layers=AUDIO_LSTM_LAYERS,
            dropout=AUDIO_LSTM_DROPOUT if AUDIO_LSTM_LAYERS > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(AUDIO_LSTM_HIDDEN * 2, 256)
        self.dropout = nn.Dropout(AUDIO_LSTM_DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N_MFCC, T] MFCC特征
        返回: [B, 256]
        """
        feat = self.conv(x)                           # [B, 128, T//4]
        feat = feat.transpose(1, 2)                   # [B, T//4, 128]
        feat = self.dropout(feat)
        lstm_out, (h_n, _) = self.lstm(feat)          # [B, T//4, H*2]
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h = torch.cat([h_forward, h_backward], dim=-1)
        return self.proj(h)                           # [B, 256]


# ==================== 跨模态注意力融合层 ====================
class CrossModalAttentionFusion(nn.Module):
    """
    将三模态特征堆叠为序列，使用多头自注意力学习模态间交互。
    输入: [h_text; h_img; h_audio], 每个 [B, 256]
    输出: [B, 512]
    """

    def __init__(self, d_model: int = 256, n_heads: int = FUSION_ATTENTION_HEADS):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=FUSION_DROPOUT,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 3, FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(FUSION_DROPOUT),
        )

    def forward(self, h_text: torch.Tensor, h_img: torch.Tensor,
                h_audio: torch.Tensor) -> torch.Tensor:
        """
        三模态 → 堆叠 → 自注意力 → 融合 → [B, 512]
        """
        # 堆叠为序列: [B, 3, 256]
        stacked = torch.stack([h_text, h_img, h_audio], dim=1)

        # 多头自注意力
        attn_out, attn_weights = self.attention(stacked, stacked, stacked)
        # Residual connection
        fused_seq = self.norm(stacked + attn_out)     # [B, 3, 256]

        # 展平 + FC → 融合向量
        fused_flat = fused_seq.reshape(fused_seq.size(0), -1)  # [B, 768]
        h_fused = self.fc(fused_flat)                           # [B, 512]

        return h_fused, attn_weights


# ==================== 完整模型 ====================
class MultiModalEmotionNet(nn.Module):
    """多模态情感识别完整模型"""

    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.audio_encoder = AudioEncoder()
        self.fusion = CrossModalAttentionFusion()
        self.classifier = nn.Sequential(
            nn.Linear(FUSION_HIDDEN_DIM, CLASSIFIER_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, NUM_CLASSES),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, text: torch.Tensor, image: torch.Tensor,
                audio: torch.Tensor) -> tuple:
        """
        前向传播
        Args:
            text:  [B, max_len]       token ids
            image: [B, 1, 48, 48]     人脸灰度图
            audio: [B, N_MFCC, T]     MFCC特征
        Returns:
            logits: [B, NUM_CLASSES]  预测logits
            attn_weights: [B, 3, 3]  模态间注意力权重 (用于可视化)
        """
        h_text = self.text_encoder(text)
        h_img = self.image_encoder(image)
        h_audio = self.audio_encoder(audio)

        h_fused, attn_weights = self.fusion(h_text, h_img, h_audio)
        logits = self.classifier(h_fused)

        return logits, attn_weights

    def forward_single_modality(self, text=None, image=None, audio=None) -> torch.Tensor:
        """单模态推理（缺失模态用零向量替代）"""
        B = 1
        device = next(self.parameters()).device
        h_text = torch.zeros(B, 256, device=device)
        h_img = torch.zeros(B, 256, device=device)
        h_audio = torch.zeros(B, 256, device=device)

        if text is not None:
            h_text = self.text_encoder(text.to(device))
        if image is not None:
            h_img = self.image_encoder(image.to(device))
        if audio is not None:
            h_audio = self.audio_encoder(audio.to(device))

        h_fused, attn_weights = self.fusion(h_text, h_img, h_audio)
        logits = self.classifier(h_fused)
        return logits, attn_weights

    def count_parameters(self) -> dict:
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # 测试模型
    model = MultiModalEmotionNet()
    params = model.count_parameters()
    print(f"MultiModalEmotionNet: {params['total']:,} total params, "
          f"{params['trainable']:,} trainable")

    # 测试前向传播
    B = 4
    text = torch.randint(0, 5000, (B, MAX_TEXT_LEN))
    image = torch.randn(B, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    audio = torch.randn(B, N_MFCC, MAX_AUDIO_FRAMES)

    logits, attn = model(text, image, audio)
    print(f"Input: text={text.shape}, image={image.shape}, audio={audio.shape}")
    print(f"Output: logits={logits.shape}, attention={attn.shape}")
    print(f"Predictions: {torch.argmax(logits, dim=-1)}")
