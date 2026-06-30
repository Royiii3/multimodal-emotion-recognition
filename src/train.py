"""
训练入口脚本
1. 生成虚拟数据或加载预处理数据
2. 创建DataLoader
3. 初始化模型和训练器
4. 开始训练
"""

import os
import sys
import json
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.dataset import MultiModalEmotionDataset, create_dataloaders
from src.model import MultiModalEmotionNet
from src.trainer import EmotionTrainer


def seed_everything(seed: int = 42):
    """固定随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_data_ready() -> bool:
    """检查预处理数据是否就绪"""
    required = [
        os.path.join(DATA_DIR, "train_samples.json"),
        os.path.join(DATA_DIR, "val_samples.json"),
        os.path.join(DATA_DIR, "test_samples.json"),
        os.path.join(DATA_DIR, "text_vocab.pkl"),
    ]
    return all(os.path.exists(f) for f in required)


def generate_demo_data():
    """
    生成演示数据（当真实数据集不可用时）
    创建 600 条三模态样本用于快速验证模型流程
    """
    print("\n[WARN] 未找到预处理数据，自动生成演示数据集...")
    print("   (正式训练前请运行: python src/preprocess.py)\n")

    # 文本生成
    texts = [
        "i am so happy today everything is wonderful",
        "this makes me really angry and frustrated",
        "i feel so sad and lonely right now",
        "i am terrified of what might happen next",
        "that is absolutely disgusting i cannot stand it",
        "what a wonderful surprise this is amazing",
        "i dont have any particular feelings about this",
        "i feel calm and peaceful right now",
        "laughing with my friends is the best feeling ever",
        "i will not tolerate this kind of disrespect anymore",
        "a deep sorrow filled my heart when i heard the news",
        "the dark shadows made me extremely nervous and scared",
        "the rotten smell was utterly revolting and gross",
        "wow i did not see that coming at all incredible",
        "it is just another ordinary day nothing special",
        "pure bliss and happiness overflow from my heart",
        "get out of my face right now i am furious",
        "my heart aches with grief and sadness everyday",
        "please dont hurt me i am so scared right now",
        "throw that nasty disgusting thing away immediately",
        "i am truly blessed and thankful for everything",
        "you are making me so mad right now stop it",
        "the loneliness is overwhelming and painful",
        "something terrible is about to happen i can feel it",
        "this garbage smells absolutely awful and nasty",
        "oh my god are you serious that is shocking",
        "just a normal tuesday afternoon at the office",
        "my heart is full of joy and gratitude today",
        "i could barely control my fury and rage",
        "tears fell down my face as i remembered the past",
    ]
    emotion_labels = [
        "happy", "angry", "sad", "fearful", "disgust", "surprised",
        "neutral", "happy", "angry", "sad", "fearful",
        "disgust", "surprised", "neutral", "happy", "angry", "sad",
        "fearful", "disgust", "happy", "angry", "sad", "fearful",
        "disgust", "surprised", "neutral", "happy", "angry", "sad",
    ]

    # 构建样本
    def make_samples(count):
        samples = []
        for i in range(count):
            idx = i % len(texts)
            samples.append({
                "text": texts[idx],
                "image_path": None,   # 触发零图像（无人脸数据）
                "image_data": None,
                "audio_path": f"synth_audio_{i}.pt",
                "label": emotion_labels[idx],
            })
        return samples

    n = 600
    train_n = int(n * TRAIN_RATIO)
    val_n = int(n * VAL_RATIO)
    test_n = n - train_n - val_n

    random.seed(42)
    all_samples = make_samples(n)
    random.shuffle(all_samples)

    splits = {
        "train": all_samples[:train_n],
        "val": all_samples[train_n:train_n + val_n],
        "test": all_samples[train_n + val_n:],
    }

    # 保存
    for split_name, split_data in splits.items():
        path = os.path.join(DATA_DIR, f"{split_name}_samples.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_data)} samples → {path}")

    # 保存词表（简单版本）
    from src.preprocess import TextPreprocessor
    tp = TextPreprocessor()
    all_texts = [s["text"] for s in all_samples]
    tp.build_vocab(all_texts)
    tp.save(os.path.join(DATA_DIR, "text_vocab.pkl"))
    print(f"  Vocab: {tp.vocab_size} words")

    # 生成随机MFCC文件
    preprocessed_dir = os.path.join(DATA_DIR, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    for i in range(n):
        mfcc = torch.randn(N_MFCC, MAX_AUDIO_FRAMES) * 0.5
        torch.save(mfcc, os.path.join(preprocessed_dir, f"synth_audio_{i}.pt"))

    print(f"\n[OK] Demo data ready: {n} samples, {tp.vocab_size} vocab words")


def main():
    print_config()

    # 固定随机种子
    seed_everything(42)

    # 检查数据
    if not check_data_ready():
        generate_demo_data()

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders()

    # 创建模型
    model = MultiModalEmotionNet()
    params = model.count_parameters()
    print(f"\n[MODEL] Model: {params['total']:,} total params ({params['trainable']:,} trainable)")

    # 打印模型结构
    print("\n" + "=" * 60)
    print("Model Architecture:")
    print("=" * 60)
    print(model)
    print("=" * 60)

    # 创建训练器
    trainer = EmotionTrainer(model)

    # 开始训练
    trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)

    # 测试集评估
    print("\n" + "=" * 60)
    print("[CHART] Final Test Set Evaluation")
    print("=" * 60)
    trainer.model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"].to(DEVICE)
            image = batch["image"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits, _ = trainer.model(text, image, audio)
            loss = trainer.criterion(logits, labels)
            test_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = test_correct / test_total
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"Test Acc:  {test_acc:.2%}")

    # 分类报告
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n[LIST] Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=EMOTION_LABELS,
        zero_division=0,
    ))

    # 保存报告
    report = classification_report(
        all_labels, all_preds,
        target_names=EMOTION_LABELS,
        zero_division=0,
    )
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), 'w') as f:
        f.write(report)
    print(f"\n[FILE] Report saved to {RESULTS_DIR}/classification_report.txt")

    # 混淆矩阵
    import matplotlib.pyplot as plt
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    ax.set_yticklabels(EMOTION_LABELS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im)

    # 标注数字
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        fontsize=8, color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[CHART] Confusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")

    print("\n[OK] Training pipeline complete!")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()
