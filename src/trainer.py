"""
多模态情感识别训练器
- AMP混合精度训练
- 梯度累积
- 标签平滑
- 学习率调度 (CosineAnnealingWarmRestarts + Warmup)
- Early Stopping
- Checkpoint保存/恢复
- 训练历史记录与绘图
"""

import os
import sys
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.model import MultiModalEmotionNet


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class EmotionTrainer:
    """多模态情感识别训练器"""

    def __init__(self, model: MultiModalEmotionNet = None):
        self.model = model if model else MultiModalEmotionNet()
        self.model = self.model.to(DEVICE)

        # 损失函数 (标签平滑)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            weight_decay=WEIGHT_DECAY,
        )

        # 混合精度
        self.scaler = GradScaler() if USE_AMP else None

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=LR_T_0, T_mult=LR_T_MULT, eta_min=MIN_LR
        )

        # 训练历史
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }

        # Early Stopping
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.current_epoch = 0

    def train_epoch(self, train_loader) -> dict:
        """训练一个epoch，返回平均loss和acc"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for step, batch in enumerate(pbar):
            text = batch["text"].to(DEVICE)
            image = batch["image"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # 前向传播
            if USE_AMP:
                with autocast():
                    logits, _ = self.model(text, image, audio)
                    loss = self.criterion(logits, labels)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                self.scaler.scale(loss).backward()
            else:
                logits, _ = self.model(text, image, audio)
                loss = self.criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

            # 梯度累积
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # 统计
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{total_loss / (step + 1):.4f}",
                "acc": f"{correct / total:.4f}",
            })

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        return {"loss": avg_loss, "acc": avg_acc}

    @torch.no_grad()
    def validate_epoch(self, val_loader) -> dict:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]  ")
        for batch in pbar:
            text = batch["text"].to(DEVICE)
            image = batch["image"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits, _ = self.model(text, image, audio)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss / (pbar.n + 1):.4f}",
                "acc": f"{correct / total:.4f}",
            })

        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total
        return {"loss": avg_loss, "acc": avg_acc}

    def fit(self, train_loader, val_loader, max_epochs: int = MAX_EPOCHS):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"[START] Training on {DEVICE}")
        print(f"   AMP: {USE_AMP} | Batch: {BATCH_SIZE}×{GRADIENT_ACCUMULATION_STEPS}")
        print(f"   LR: {LEARNING_RATE} → {MIN_LR} | Patience: {EARLY_STOPPING_PATIENCE}")
        print(f"{'='*60}\n")

        # Warmup
        if WARMUP_EPOCHS > 0:
            warmup_lr = LEARNING_RATE * 0.1
            for pg in self.optimizer.param_groups:
                pg['lr'] = warmup_lr
            print(f"[WARM] Warmup: lr={warmup_lr:.2e} for {WARMUP_EPOCHS} epochs")

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # --- Train ---
            train_metrics = self.train_epoch(train_loader)

            # --- Validate ---
            val_metrics = self.validate_epoch(val_loader)

            # --- LR Step ---
            if epoch >= WARMUP_EPOCHS:
                self.scheduler.step()
            else:
                # Linear warmup
                progress = (epoch + 1) / WARMUP_EPOCHS
                lr = warmup_lr + (LEARNING_RATE - warmup_lr) * progress
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time

            # --- 记录 ---
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["lr"].append(current_lr)

            # --- 打印 ---
            print(f"\n[CHART] Epoch {epoch+1}/{max_epochs} | Time: {elapsed:.1f}s | LR: {current_lr:.2e}")
            print(f"   Train ─ Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.2%}")
            print(f"   Val   ─ Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.2%}")

            # --- Checkpoint ---
            val_acc = val_metrics["acc"]
            val_loss = val_metrics["loss"]

            # 按最高acc保存
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(BEST_MODEL_PATH)
                print(f"   [OK] Best model saved! (val_acc={val_acc:.2%})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"   [WAIT] No improvement ({self.patience_counter}/{EARLY_STOPPING_PATIENCE})")

            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n[STOP] Early stopping at epoch {epoch+1}")
                break

        # --- 保存最终模型 ---
        self._save_checkpoint(FINAL_MODEL_PATH)
        print(f"\n[SAVE] Final model saved: {FINAL_MODEL_PATH}")
        print(f"[BEST] Best val_acc: {self.best_val_acc:.2%}")

        # --- 绘制训练曲线 ---
        self._plot_training_curves()

    def _save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "config": {
                "num_classes": NUM_CLASSES,
                "emotion_labels": EMOTION_LABELS,
            },
        }
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.current_epoch = checkpoint.get("epoch", 0)
        print(f"[LOAD] Loaded checkpoint from {path} (epoch {self.current_epoch+1})")

    def _plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history["train_acc"], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.history["val_acc"], 'r-', label='Val Acc', linewidth=2)
        axes[1].axhline(y=self.best_val_acc, color='g', linestyle='--',
                        label=f'Best: {self.best_val_acc:.2%}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # LR
        axes[2].plot(epochs, self.history["lr"], 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "training_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[PLOT] Training curves saved: {path}")


if __name__ == "__main__":
    # 快速测试
    print("Trainer module loaded successfully.")
    model = MultiModalEmotionNet()
    trainer = EmotionTrainer(model)
    print(f"Model params: {model.count_parameters()}")
