# Multimodal Emotion Recognition · 多模态情感识别

基于深度学习的**文本 + 人脸图像**双模态情感识别系统。使用 PyTorch 从零搭建，Streamlit 部署。

## 模型

| 模型 | 架构 | 参数量 | 验证准确率 | 数据来源 |
|------|------|--------|-----------|----------|
| 文本 | BiLSTM (2层, 双向) | 4M | 94.81% | Kaggle 20K 情感文本 |
| 人脸图像 | 4层 CNN | 5.6M | 87%+ | FER2013 (35K 灰度人脸) |

两个模型**独立训练、独立推理**，通过 Streamlit 前端统一展示结果。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型（可选 — 已提供预训练权重）
python src/image_trainer.py    # 训练图像模型
python src/text_trainer.py     # 训练文本模型

# 3. 启动 Web 应用
streamlit run app.py
```

## 项目结构

```
├── app.py                     # Streamlit 前端界面
├── requirements.txt           # Python 依赖
├── README.md
├── src/
│   ├── config.py              # 超参数 & 路径配置
│   ├── preprocess.py          # 文本/图像/音频预处理
│   ├── dataset.py             # Dataset & DataLoader
│   ├── model.py               # 多模态融合模型（三编码器 + 注意力）
│   ├── image_model.py         # 独立图像 CNN 分类器
│   ├── text_model.py          # 独立文本 BiLSTM 分类器
│   ├── trainer.py             # 融合模型训练器
│   ├── image_trainer.py       # 图像模型训练器
│   ├── text_trainer.py        # 文本模型训练器
│   ├── train.py               # 训练入口（融合模型）
│   ├── inference.py           # 双模型推理（app.py 调用）
│   ├── build_final_dataset.py # 数据集构建
│   └── build_texts.py         # 文本语料构建
├── models/                    # 模型权重
│   ├── best_image_model.pth   # 图像分类器（760 KB）
│   └── best_text_model.pth    # 文本分类器（16 MB）
├── data/                      # 数据集（gitignored — 需自行准备）
│   ├── text_vocab.pkl         # 文本词表（推理必须）
│   ├── *_samples.json         # 训练/验证/测试索引
│   └── all_images.npy         # 人脸图像数组（FER2013）
└── results/                   # 训练曲线 & 混淆矩阵
```

## 支持的 emotion 类别

`neutral` · `happy` · `sad` · `angry` · `fearful` · `disgust` · `surprised`

## 部署到 Streamlit Cloud

1. Fork/Push 本仓库到 GitHub
2. 打开 [share.streamlit.io](https://share.streamlit.io)
3. 连接 GitHub 仓库，设置 `app.py` 为主文件
4. 点击 Deploy — 模型权重已在仓库中，无需额外配置

> **注意**：`data/` 下的 `.npy` 和 `.json` 文件已被 gitignore。Streamlit Cloud 部署仅需模型权重（已在 `models/` 中追踪）。如需本地训练，请自行下载 FER2013 数据集并运行 `src/image_trainer.py`。
