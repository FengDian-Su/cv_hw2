"""
train.py - DETR ResNet-50 Digit Detection
專案結構:
    cv_hw2/
    ├── code/train.py
    ├── data/
    │   ├── train/      (圖片)
    │   ├── valid/      (圖片)
    │   ├── test/       (圖片)
    │   ├── train.json
    │   └── valid.json
    └── model/          (checkpoint 儲存位置)
"""

import os
import json
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from PIL import Image


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
DEVICE        = torch.device('cuda:0')
DATA_DIR      = './data'
MODEL_DIR     = './model'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
VALID_IMG_DIR = os.path.join(DATA_DIR, 'valid')
TRAIN_JSON    = os.path.join(DATA_DIR, 'train.json')
VALID_JSON    = os.path.join(DATA_DIR, 'valid.json')
LOG_PATH      = os.path.join(MODEL_DIR, 'train_log.csv')

BATCH_SIZE    = 8
NUM_EPOCHS    = 50
LR            = 1e-4
LR_BACKBONE   = 1e-5
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 0.1
NUM_WORKERS   = 4
SAVE_EVERY    = 5     # 每幾個 epoch 存一次 checkpoint
PRETRAINED    = 'facebook/detr-resnet-50'
LOCAL_CONFIG  = 'model/detr-resnet-50'

# category_id 1~10 對應數字 '0'~'9'
ID2LABEL = {i: str(i - 1) for i in range(1, 11)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────
class CSVLogger:
    def __init__(self, path):
        self.path = path
        # 如果檔案不存在，寫入 header
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch',
                    'train_loss', 'train_ce', 'train_bbox', 'train_giou',
                    'val_loss'
                ])

    def log(self, epoch, train_metrics, val_loss):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_metrics['loss']:.6f}",
                f"{train_metrics['loss_ce']:.6f}",
                f"{train_metrics['loss_bbox']:.6f}",
                f"{train_metrics['loss_giou']:.6f}",
                f"{val_loss:.6f}",
            ])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class DigitDetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_file, processor):
        self.img_dir   = img_dir
        self.processor = processor

        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.images  = {img['id']: img for img in coco['images']}
        self.img_ids = [img['id'] for img in coco['images']]

        # image_id -> list of annotations
        self.annotations = {img_id: [] for img_id in self.img_ids}
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image    = Image.open(img_path).convert('RGB')

        target = {
            'image_id':    img_id,
            'annotations': self.annotations[img_id],
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors='pt'
        )

        return (
            encoding['pixel_values'].squeeze(0),   # (3, H, W)
            encoding['labels'][0]                  # dict of tensors
        )


# ─────────────────────────────────────────────
# Collate fn（處理不同數量的 bbox）
# ─────────────────────────────────────────────
def make_collate_fn(processor):
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        labels       = [item[1] for item in batch]

        # 找這個 batch 裡最大的 H 和 W
        max_h = max(p.shape[1] for p in pixel_values)
        max_w = max(p.shape[2] for p in pixel_values)

        padded_images = []
        pixel_masks   = []

        for p in pixel_values:
            c, h, w = p.shape
            
            # 右下角補零
            pad = torch.zeros(c, max_h, max_w)
            pad[:, :h, :w] = p
            padded_images.append(pad)
            
            # mask：有圖片的地方是 1，padding 的地方是 0
            mask = torch.zeros(max_h, max_w, dtype=torch.long)
            mask[:h, :w] = 1
            pixel_masks.append(mask)

        return {
            'pixel_values': torch.stack(padded_images),   # (B, 3, H, W)
            'pixel_mask':   torch.stack(pixel_masks),     # (B, H, W)
            'labels':       labels,
        }
    return collate_fn


# ─────────────────────────────────────────────
# 訓練一個 epoch
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss      = 0.0
    total_loss_ce   = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    n_batches       = len(loader)
    t0 = time.time()

    for step, batch in enumerate(loader):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask   = batch['pixel_mask'].to(device)
        labels       = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss      += loss.item()
        total_loss_ce   += outputs.loss_dict.get('loss_ce',   torch.tensor(0.0)).item()
        total_loss_bbox += outputs.loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
        total_loss_giou += outputs.loss_dict.get('loss_giou', torch.tensor(0.0)).item()

        if (step + 1) % 50 == 0 or (step + 1) == n_batches:
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch} | Step {step+1}/{n_batches}] "
                  f"loss={loss.item():.4f}  "
                  f"ce={outputs.loss_dict.get('loss_ce',   torch.tensor(0.0)).item():.4f}  "
                  f"bbox={outputs.loss_dict.get('loss_bbox', torch.tensor(0.0)).item():.4f}  "
                  f"giou={outputs.loss_dict.get('loss_giou', torch.tensor(0.0)).item():.4f}  "
                  f"({elapsed:.1f}s)")

    return {
        'loss':      total_loss      / n_batches,
        'loss_ce':   total_loss_ce   / n_batches,
        'loss_bbox': total_loss_bbox / n_batches,
        'loss_giou': total_loss_giou / n_batches,
    }


# ─────────────────────────────────────────────
# 驗證一個 epoch
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, epoch):
    model.eval()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch in loader:
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask   = batch['pixel_mask'].to(device)
        labels       = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )
        total_loss += outputs.loss.item()

    avg_loss = total_loss / n_batches
    print(f"  [Epoch {epoch} | Val] loss={avg_loss:.4f}")
    return avg_loss


# ─────────────────────────────────────────────
# Checkpoint 儲存 / 讀取
# ─────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss':             val_loss,
    }, path)
    print(f"  → Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"  → Checkpoint loaded: {path}  (epoch {ckpt['epoch']})")
    return ckpt['epoch'], ckpt['val_loss']


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"Device : {DEVICE}")
    print(f"Epochs : {NUM_EPOCHS}  |  Batch : {BATCH_SIZE}")
    print("=" * 60)

    processor = DetrImageProcessor.from_pretrained(PRETRAINED, revision='no_timm')

    train_dataset = DigitDetectionDataset(TRAIN_IMG_DIR, TRAIN_JSON, processor)
    valid_dataset = DigitDetectionDataset(VALID_IMG_DIR, VALID_JSON, processor)
    collate_fn    = make_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Valid: {len(valid_dataset)} images, {len(valid_loader)} batches")

    # 先從 pretrained 抓 config，手動修正 dilation
    config = DetrConfig.from_pretrained(LOCAL_CONFIG)
    config.num_labels = 10
    config.id2label   = ID2LABEL
    config.label2id   = LABEL2ID

    # ── Model：backbone 用 pretrained，encoder/decoder 從頭訓練 ──
    model = DetrForObjectDetection(config)  # 全部隨機初始化

    # 只把 backbone 的 pretrained 權重載入
    pretrained_full = DetrForObjectDetection.from_pretrained(
        PRETRAINED,
        revision='no_timm',
        config=config,
        ignore_mismatched_sizes=True,
    )

    backbone_state = {
        k: v for k, v in pretrained_full.state_dict().items()
        if 'backbone' in k
    }
    model.load_state_dict(backbone_state, strict=False)
    del pretrained_full
    torch.cuda.empty_cache()

    model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.1f}M")

    # ── Optimizer（backbone 用較小的 lr）──
    backbone_params     = [p for n, p in model.named_parameters() if 'backbone' in n]
    non_backbone_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    optimizer = torch.optim.AdamW([
        {'params': non_backbone_params, 'lr': LR},
        {'params': backbone_params,     'lr': LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)

    # ── Logger ──
    logger = CSVLogger(LOG_PATH)
    print(f"Logging to: {LOG_PATH}")

    start_epoch   = 1
    best_val_loss = float('inf')
    resume_path   = os.path.join(MODEL_DIR, 'latest.pth')
    if os.path.exists(resume_path):
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, resume_path, DEVICE)
        start_epoch += 1

    # ── 訓練迴圈 ──
    print("\nStart training...\n")
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'─' * 60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss      = validate(model, valid_loader, DEVICE, epoch)

        print(f"  Train → loss={train_metrics['loss']:.4f}  "
              f"ce={train_metrics['loss_ce']:.4f}  "
              f"bbox={train_metrics['loss_bbox']:.4f}  "
              f"giou={train_metrics['loss_giou']:.4f}")

        # ── 寫入 log ──
        logger.log(epoch, train_metrics, val_loss)

        save_checkpoint(model, optimizer, epoch, val_loss,
                        os.path.join(MODEL_DIR, 'latest.pth'))

        # 最佳 checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(MODEL_DIR, 'best.pth'))
            print(f"  ★ New best val loss: {best_val_loss:.4f}")

        # 定期 checkpoint
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(MODEL_DIR, f'epoch_{epoch:03d}.pth'))

    print("\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best model    : {os.path.join(MODEL_DIR, 'best.pth')}")
    print(f"Log saved at  : {LOG_PATH}")

if __name__ == '__main__':
    main()