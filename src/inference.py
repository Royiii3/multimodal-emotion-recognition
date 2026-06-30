"""
Inference — two independent models: Text + Image
No fusion, no audio. Clean and simple.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.model import MultiModalEmotionNet
from src.text_model import TextEmotionClassifier
from src.preprocess import TextPreprocessor


class EmotionPredictor:
    """Dual-model emotion recognition: text classifier + image classifier"""

    def __init__(self):
        self.device = DEVICE

        # --- Image model ---
        self.image_model = MultiModalEmotionNet().to(self.device)
        img_path = os.path.join(MODEL_DIR, "best_model.pth")
        if os.path.exists(img_path):
            ckpt = torch.load(img_path, map_location=self.device, weights_only=False)
            self.image_model.load_state_dict(ckpt["model_state_dict"])
            self.image_model.eval()
            print(f"[OK] Image model loaded (best_val_acc={ckpt.get('best_val_acc',0):.2%})")
        else:
            print(f"[WARN] Image model not found at {img_path}")

        # --- Text model ---
        text_path = os.path.join(MODEL_DIR, "best_text_model.pth")
        if os.path.exists(text_path):
            ckpt = torch.load(text_path, map_location=self.device, weights_only=False)
            vocab_size = ckpt.get("vocab_size", MAX_VOCAB_SIZE)
            self.text_model = TextEmotionClassifier(vocab_size=vocab_size).to(self.device)
            self.text_model.load_state_dict(ckpt["model_state_dict"])
            self.text_model.eval()
            print(f"[OK] Text model loaded (best_val_acc={ckpt.get('best_val_acc',0):.2%})")
        else:
            print(f"[WARN] Text model not found at {text_path}")
            self.text_model = None

        # --- Text processor ---
        self.text_processor = None
        vocab_path = os.path.join(DATA_DIR, "text_vocab.pkl")
        if os.path.exists(vocab_path):
            self.text_processor = TextPreprocessor()
            self.text_processor.load(vocab_path)

    # ==================== Face detection ====================

    def detect_faces(self, image_bgr: np.ndarray):
        """Detect all faces using Haar Cascade."""
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = image_bgr if len(image_bgr.shape) == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

        results = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
                face = face.astype(np.float32) / 255.0
                face = (face - IMAGE_MEAN) / IMAGE_STD
                results.append((face, (x, y, w, h)))
        else:
            full = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
            full = full.astype(np.float32) / 255.0
            full = (full - IMAGE_MEAN) / IMAGE_STD
            results.append((full, None))
        return results

    # ==================== Predict ====================

    def predict_text(self, text: str) -> dict:
        """Text-only prediction"""
        if not self.text_model or not self.text_processor:
            return {"error": f"Text model ({self.text_model is not None}) / vocab ({self.text_processor is not None})"}
            return {"error": "Text model not loaded"}
        tokens = self.text_processor.encode(text).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.text_model(tokens)
        return self._format(logits, "text")

    def predict_image(self, face_np: np.ndarray) -> dict:
        """Image-only prediction (face_np: 48x48 preprocessed array)"""
        img = torch.tensor(face_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        txt = torch.zeros(1, MAX_TEXT_LEN, dtype=torch.long).to(self.device)
        aud = torch.zeros(1, N_MFCC, MAX_AUDIO_FRAMES).to(self.device)
        with torch.no_grad():
            logits, _ = self.image_model.forward_single_modality(text=txt, image=img, audio=aud)
        return self._format(logits, "image")

    def _format(self, logits: torch.Tensor, modality: str) -> dict:
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        top3_idx = np.argsort(probs)[::-1][:3]
        return {
            "prediction": EMOTION_LABELS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "top3": [{"label": EMOTION_LABELS[i], "confidence": float(probs[i])} for i in top3_idx],
            "all_probs": {EMOTION_LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)},
            "modality": modality,
        }
