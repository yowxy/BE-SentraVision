"""
SentraVision: Dataset Ingestion and Fine-Tuning Pipeline
Downloads/extracts the Kaggle SentraVision dataset and runs model fine-tuning.
"""

import os
import sys
import glob
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = Path("./dataset_minimarket")
DOWNLOADS_DIR = Path("/home/iklil/Downloads")
CLASSES = ["Normal Shopping", "Conceal in Pocket", "Bag Shoplifting", "Loitering/Casing"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

SEQUENCE_LENGTH = 16
FRAME_SIZE = (112, 112)
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cari_dan_ekstrak_dataset():
    """
    Mencari file zip dataset di folder Downloads atau memicu unduhan Kaggle.
    """
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Cek apakah dataset sudah diekstrak
    existing_videos = list(BASE_DIR.rglob("*.mp4")) + list(BASE_DIR.rglob("*.avi"))
    if len(existing_videos) > 0:
        print(f"Dataset lokal sudah tersedia: {len(existing_videos)} video terdeteksi.")
        return True

    # 2. Cek apakah ada file zip di Downloads
    candidate_zips = list(DOWNLOADS_DIR.glob("*sentravision*.zip")) + list(DOWNLOADS_DIR.glob("*sentra*.zip"))
    if candidate_zips:
        zip_path = candidate_zips[0]
        print(f"Menemukan file zip dataset di: {zip_path}")
        print("Mengekstrak dataset ke folder lokal dataset_minimarket...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        print("Ekstraksi selesai.")
        return True

    # 3. Cek apakah ada kredensial kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print("Kredensial Kaggle terdeteksi. Memulai unduhan via Kaggle API...")
        os.system("kaggle datasets download -d hanestyan/sentravision-datasets -p ./dataset_minimarket --unzip")
        return True

    print("Kaggle membutuhkan autentikasi akun untuk mengunduh dataset.")
    print("Silakan unduh file zip dari tautan: https://www.kaggle.com/datasets/hanestyan/sentravision-datasets")
    print("Letakkan file zip di folder Downloads atau di dataset_minimarket.")
    return False


class MinimarketVideoDataset(Dataset):
    def __init__(self, root_dir: Path, sequence_length=16, frame_size=(112, 112)):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.video_paths = []
        self.labels = []
        
        # Scan folder per kelas
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_path = root_dir / cls_name
            if cls_path.exists():
                for ext in ["*.mp4", "*.avi", "*.mkv"]:
                    for v_file in cls_path.glob(ext):
                        self.video_paths.append(v_file)
                        self.labels.append(cls_idx)
                        
        # Jika struktur flat atau subfolder lain
        if len(self.video_paths) == 0:
            for ext in ["*.mp4", "*.avi", "*.mkv"]:
                for v_file in root_dir.rglob(ext):
                    # Deteksi label dari nama folder atau file
                    fname_lower = v_file.stem.lower() + " " + v_file.parent.name.lower()
                    if "conceal" in fname_lower or "pocket" in fname_lower or "curi" in fname_lower:
                        lbl = 1
                    elif "bag" in fname_lower or "tas" in fname_lower:
                        lbl = 2
                    elif "loiter" in fname_lower or "diam" in fname_lower:
                        lbl = 3
                    else:
                        lbl = 0
                    self.video_paths.append(v_file)
                    self.labels.append(lbl)

    def __len__(self):
        return len(self.video_paths)

    def _load_uniform_frames(self, video_path: Path):
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.sequence_length
            
        indices = np.linspace(0, max(0, total_frames - 1), self.sequence_length, dtype=int)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
        cap.release()
        
        while len(frames) < self.sequence_length:
            frames.append(np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8))
            
        return np.array(frames[:self.sequence_length])

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        raw_frames = self._load_uniform_frames(video_path)
        tensor_frames = torch.tensor(raw_frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        return tensor_frames, torch.tensor(label, dtype=torch.long)


class LRCN_BiLSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_dim=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x_flat)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context_vector)


def jalankan_finetuning():
    dataset_ready = cari_dan_ekstrak_dataset()
    if not dataset_ready:
        print("Menggunakan video lokal yang tersedia di dataset_minimarket...")

    full_dataset = MinimarketVideoDataset(BASE_DIR, sequence_length=SEQUENCE_LENGTH, frame_size=FRAME_SIZE)
    if len(full_dataset) == 0:
        print("Tidak ada video yang ditemukan. Menyiapkan dataset demo awal...")
        os.system("python3 -c 'from create_clean_notebook import *'")
        full_dataset = MinimarketVideoDataset(BASE_DIR, sequence_length=SEQUENCE_LENGTH, frame_size=FRAME_SIZE)

    print(f"Total video yang digunakan untuk fine-tuning: {len(full_dataset)} video.")
    
    # Train validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LRCN_BiLSTM(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("Memulai proses fine-tuning model...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_videos, batch_labels in train_loader:
            batch_videos = batch_videos.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_videos)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_videos.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            
        train_acc = (correct / total) * 100 if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Akurasi Train: {train_acc:.2f}%")

    # Evaluasi pada data validasi
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_videos, batch_labels in val_loader:
            batch_videos = batch_videos.to(device)
            outputs = model(batch_videos)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_labels.numpy())

    if len(all_targets) > 0:
        val_acc = accuracy_score(all_targets, all_preds) * 100
        val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100
        print(f"\nHasil Evaluasi Fine-Tuning pada Data Validasi:")
        print(f"- Akurasi Validasi: {val_acc:.2f}%")
        print(f"- Macro F1-Score: {val_macro_f1:.2f}%")
    
    # Simpan bobot model hasil fine-tuning
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/sentravision_finetuned.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model hasil fine-tuning berhasil disimpan pada: {model_save_path}")


if __name__ == "__main__":
    jalankan_finetuning()
