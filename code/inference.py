"""
inference.py - DETR ResNet-50 Digit Detection
功能：
  1. 對 validation set 跑推論，用 pycocotools 計算 mAP
  2. 對 test set 跑推論，產生 pred.json 供上傳 CodaBench
  3. 產生各種視覺化圖表

執行方式：
  CUDA_VISIBLE_DEVICES=1 python code/inference.py --mode val --threshold 0.3
  CUDA_VISIBLE_DEVICES=1 python code/inference.py --mode test --threshold 0.3
  CUDA_VISIBLE_DEVICES=1 python code/inference.py --mode both --threshold 0.3
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
DEVICE        = torch.device('cuda:0')
DATA_DIR      = 'data'
MODEL_DIR     = 'model'
VIS_DIR       = 'visualizations'
VALID_IMG_DIR = os.path.join(DATA_DIR, 'valid')
TEST_IMG_DIR  = os.path.join(DATA_DIR, 'test')
VALID_JSON    = os.path.join(DATA_DIR, 'valid.json')
CHECKPOINT    = os.path.join(MODEL_DIR, 'best.pth')
PRETRAINED    = 'facebook/detr-resnet-50'
LOG_PATH      = os.path.join(MODEL_DIR, 'train_log.csv')

BATCH_SIZE     = 8
NUM_WORKERS    = 4
CONF_THRESHOLD = 0.5

ID2LABEL = {i: str(i - 1) for i in range(1, 11)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

os.makedirs(VIS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Dataset（val）
# ─────────────────────────────────────────────
class ValDataset(Dataset):
    def __init__(self, img_dir, annotation_file, processor):
        self.img_dir   = img_dir
        self.processor = processor
        with open(annotation_file, 'r') as f:
            coco = json.load(f)
        self.images  = {img['id']: img for img in coco['images']}
        self.img_ids = [img['id'] for img in coco['images']]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]
        image    = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert('RGB')
        orig_w, orig_h = image.size
        encoding = self.processor(images=image, return_tensors='pt')
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'image_id':     img_id,
            'orig_size':    (orig_h, orig_w),
        }


# ─────────────────────────────────────────────
# Dataset（test）
# ─────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, img_dir, processor):
        self.img_dir   = img_dir
        self.processor = processor
        self.filenames = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname  = self.filenames[idx]
        image  = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        orig_w, orig_h = image.size
        encoding = self.processor(images=image, return_tensors='pt')
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'image_id':     int(os.path.splitext(fname)[0]),
            'orig_size':    (orig_h, orig_w),
        }


# ─────────────────────────────────────────────
# Collate fn
# ─────────────────────────────────────────────
def collate_fn(batch):
    pixel_values = [item['pixel_values'] for item in batch]
    image_ids    = [item['image_id']     for item in batch]
    orig_sizes   = [item['orig_size']    for item in batch]

    max_h = max(p.shape[1] for p in pixel_values)
    max_w = max(p.shape[2] for p in pixel_values)
    padded_images, pixel_masks = [], []

    for p in pixel_values:
        c, h, w = p.shape
        pad = torch.zeros(c, max_h, max_w)
        pad[:, :h, :w] = p
        padded_images.append(pad)
        mask = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:h, :w] = 1
        pixel_masks.append(mask)

    return {
        'pixel_values': torch.stack(padded_images),
        'pixel_mask':   torch.stack(pixel_masks),
        'image_ids':    image_ids,
        'orig_sizes':   orig_sizes,
    }


# ─────────────────────────────────────────────
# 載入模型
# ─────────────────────────────────────────────
def load_model(checkpoint_path):
    print(f"Loading model from: {checkpoint_path}")
    processor = DetrImageProcessor.from_pretrained(PRETRAINED, revision='no_timm')

    config = DetrConfig.from_pretrained(PRETRAINED, revision='no_timm')
    config.dilation   = False
    config.num_labels = 10
    config.id2label   = ID2LABEL
    config.label2id   = LABEL2ID

    model = DetrForObjectDetection.from_pretrained(
        PRETRAINED, revision='no_timm',
        config=config, ignore_mismatched_sizes=True,
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  → Loaded checkpoint from epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")
    model.to(DEVICE)
    model.eval()
    return model, processor


# ─────────────────────────────────────────────
# 推論
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, threshold=CONF_THRESHOLD):
    results = []
    for batch in tqdm(loader, desc='Inferring'):
        pixel_values = batch['pixel_values'].to(DEVICE)
        pixel_mask   = batch['pixel_mask'].to(DEVICE)
        image_ids    = batch['image_ids']
        orig_sizes   = batch['orig_sizes']

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.tensor(orig_sizes, dtype=torch.long).to(DEVICE)

        if hasattr(model, 'image_processor'):
            preds = model.image_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes)
        else:
            preds = _post_process(outputs, orig_sizes, threshold)

        for img_id, pred in zip(image_ids, preds):
            for score, label, box in zip(pred['scores'], pred['labels'], pred['boxes']):
                x_min, y_min, x_max, y_max = box.tolist()
                results.append({
                    'image_id':    int(img_id),
                    'category_id': int(label),
                    'bbox':        [x_min, y_min, x_max - x_min, y_max - y_min],
                    'score':       float(score),
                })
    return results


def _post_process(outputs, orig_sizes, threshold):
    import torch.nn.functional as F
    preds = []
    for logits, boxes, (orig_h, orig_w) in zip(outputs.logits, outputs.pred_boxes, orig_sizes):
        probs  = F.softmax(logits, dim=-1)[:, :-1]
        scores, labels = probs.max(dim=-1)
        cx, cy, w, h = boxes.unbind(-1)
        boxes_abs = torch.stack([
            (cx - w/2) * orig_w, (cy - h/2) * orig_h,
            (cx + w/2) * orig_w, (cy + h/2) * orig_h,
        ], dim=-1)
        keep = scores > threshold
        preds.append({'scores': scores[keep].cpu(),
                      'labels': labels[keep].cpu(),
                      'boxes':  boxes_abs[keep].cpu()})
    return preds


# ─────────────────────────────────────────────
# mAP
# ─────────────────────────────────────────────
def evaluate_map(results, annotation_file):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    print("\nCalculating mAP with pycocotools...")
    coco_gt   = COCO(annotation_file)
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP   = coco_eval.stats[0]
    mAP50 = coco_eval.stats[1]
    print(f"\n{'='*40}")
    print(f"mAP (IoU=0.50:0.95) : {mAP:.4f}")
    print(f"mAP (IoU=0.50)      : {mAP50:.4f}")
    print(f"{'='*40}")
    if mAP < 0.28:
        print(f"Competition score : 0  (below weak baseline)")
    elif mAP < 0.38:
        score = 60 + (mAP - 0.28) / (0.38 - 0.28) * 20
        print(f"Competition score : ~{score:.1f}")
    else:
        print(f"Competition score : 80+  (above strong baseline!)")
    print(f"{'='*40}\n")
    return coco_gt, coco_eval


# ═════════════════════════════════════════════
# 視覺化
# ═════════════════════════════════════════════

def plot_loss_curves(log_path=LOG_PATH):
    import csv
    if not os.path.exists(log_path):
        print(f"[Skip] Loss curve: {log_path} not found.")
        return

    epochs, train_loss, val_loss = [], [], []
    train_ce, train_bbox, train_giou = [], [], []
    with open(log_path, 'r') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            train_ce.append(float(row['train_ce']))
            train_bbox.append(float(row['train_bbox']))
            train_giou.append(float(row['train_giou']))

    # Train vs Val Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label='Train Loss', color='steelblue')
    ax.plot(epochs, val_loss,   label='Val Loss',   color='tomato', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(VIS_DIR, 'loss_curve.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # Loss breakdown
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_ce,   label='CE Loss',   color='steelblue')
    ax.plot(epochs, train_bbox, label='BBox Loss',  color='orange')
    ax.plot(epochs, train_giou, label='GIoU Loss',  color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(VIS_DIR, 'loss_breakdown.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_per_class_ap(coco_gt, coco_eval):
    precision = coco_eval.eval['precision']
    cat_ids   = sorted(coco_gt.getCatIds())
    ap_per_class = []
    for i in range(len(cat_ids)):
        p  = precision[0, :, i, 0, 2]
        ap = np.mean(p[p > -1]) if np.any(p > -1) else 0.0
        ap_per_class.append(ap)

    labels = [ID2LABEL[c] for c in cat_ids]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, ap_per_class, color='steelblue', edgecolor='white')
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('AP @ IoU=0.50')
    ax.set_title('Per-Class Average Precision (AP@IoU=0.50)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, ap_per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    out = os.path.join(VIS_DIR, 'per_class_ap.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_pr_curves(coco_gt, coco_eval):
    precision    = coco_eval.eval['precision']
    recall_thrs  = np.linspace(0, 1, 101)
    cat_ids      = sorted(coco_gt.getCatIds())
    colors       = plt.cm.tab10(np.linspace(0, 1, len(cat_ids)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (cat_id, color) in enumerate(zip(cat_ids, colors)):
        p     = precision[0, :, i, 0, 2]
        valid = p > -1
        if valid.any():
            ax.plot(recall_thrs[valid], p[valid],
                    label=f"digit {ID2LABEL[cat_id]}", color=color)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve (IoU=0.50, per digit)')
    ax.legend(loc='lower left', ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(VIS_DIR, 'pr_curves.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_confusion_matrix(results, annotation_file, iou_threshold=0.5):
    from pycocotools.coco import COCO
    coco_gt = COCO(annotation_file)
    n = 10
    cm = np.zeros((n, n), dtype=int)

    gt_by_image   = {}
    pred_by_image = {}
    for ann in coco_gt.dataset['annotations']:
        gt_by_image.setdefault(ann['image_id'], []).append(ann)
    for r in results:
        pred_by_image.setdefault(r['image_id'], []).append(r)

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = a[2]*a[3] + b[2]*b[3] - inter
        return inter / union if union > 0 else 0.0

    for img_id, gts in gt_by_image.items():
        preds   = pred_by_image.get(img_id, [])
        matched = set()
        for gt in gts:
            gt_cls = gt['category_id'] - 1
            best_iou, best_cls, best_j = 0, -1, -1
            for j, pred in enumerate(preds):
                if j in matched:
                    continue
                s = iou(gt['bbox'], pred['bbox'])
                if s > best_iou:
                    best_iou, best_cls, best_j = s, pred['category_id']-1, j
            if best_iou >= iou_threshold and best_cls >= 0:
                cm[gt_cls][best_cls] += 1
                matched.add(best_j)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    labels = [str(i) for i in range(10)]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Ground Truth')
    ax.set_title(f'Confusion Matrix (IoU ≥ {iou_threshold})')
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    out = os.path.join(VIS_DIR, 'confusion_matrix.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_prediction_samples(results, annotation_file, img_dir, n_samples=16):
    import random
    from pycocotools.coco import COCO
    coco_gt = COCO(annotation_file)

    pred_by_image = {}
    gt_by_image   = {}
    for r in results:
        pred_by_image.setdefault(r['image_id'], []).append(r)
    for ann in coco_gt.dataset['annotations']:
        gt_by_image.setdefault(ann['image_id'], []).append(ann)

    sample_infos = random.sample(coco_gt.dataset['images'],
                                 min(n_samples, len(coco_gt.dataset['images'])))
    n_cols = 4
    n_rows = (len(sample_infos) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axes = axes.flatten()

    for ax, img_info in zip(axes, sample_infos):
        img_id = img_info['id']
        image  = Image.open(os.path.join(img_dir, img_info['file_name'])).convert('RGB')
        ax.imshow(image)
        ax.set_title(f"id={img_id}", fontsize=8)
        ax.axis('off')

        for gt in gt_by_image.get(img_id, []):
            x, y, w, h = gt['bbox']
            ax.add_patch(patches.Rectangle((x, y), w, h,
                         linewidth=1, edgecolor='lime', facecolor='none'))
            ax.text(x, y-1, ID2LABEL[gt['category_id']],
                    color='lime', fontsize=7, fontweight='bold')

        for pred in pred_by_image.get(img_id, []):
            x, y, w, h = pred['bbox']
            ax.add_patch(patches.Rectangle((x, y), w, h,
                         linewidth=1, edgecolor='red',
                         facecolor='none', linestyle='--'))
            ax.text(x+w, y-1,
                    f"{ID2LABEL[pred['category_id']]}:{pred['score']:.2f}",
                    color='red', fontsize=6)

    for ax in axes[len(sample_infos):]:
        ax.axis('off')

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], color='lime', linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color='red',  linewidth=2, linestyle='--', label='Prediction'),
    ], loc='lower center', ncol=2, fontsize=10)
    plt.suptitle('Prediction Samples (Green=GT, Red=Pred)', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(VIS_DIR, 'prediction_samples.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       type=str, default='both',
                        choices=['val', 'test', 'both'])
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    parser.add_argument('--threshold',  type=float, default=CONF_THRESHOLD)
    parser.add_argument('--output',     type=str, default='pred.json')
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint)

    if args.mode in ('val', 'both'):
        print(f"\n{'─'*50}")
        print("Running inference on validation set...")
        val_dataset = ValDataset(VALID_IMG_DIR, VALID_JSON, processor)
        val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 collate_fn=collate_fn)
        print(f"Val images: {len(val_dataset)}")
        val_results = run_inference(model, val_loader, threshold=args.threshold)
        print(f"Total predictions: {len(val_results)}")

        if len(val_results) == 0:
            print("No predictions. Try lowering --threshold.")
        else:
            coco_gt, coco_eval = evaluate_map(val_results, VALID_JSON)
            print("Generating visualizations...")
            plot_loss_curves()
            plot_per_class_ap(coco_gt, coco_eval)
            plot_pr_curves(coco_gt, coco_eval)
            plot_confusion_matrix(val_results, VALID_JSON)
            plot_prediction_samples(val_results, VALID_JSON, VALID_IMG_DIR)
            print(f"\nAll visualizations saved to: {VIS_DIR}/")

    if args.mode in ('test', 'both'):
        print(f"\n{'─'*50}")
        print("Running inference on test set...")
        test_dataset = TestDataset(TEST_IMG_DIR, processor)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  collate_fn=collate_fn)
        print(f"Test images: {len(test_dataset)}")
        test_results = run_inference(model, test_loader, threshold=args.threshold)
        print(f"Total predictions: {len(test_results)}")
        with open(args.output, 'w') as f:
            json.dump(test_results, f)
        print(f"Saved: {args.output}  ({len(test_results)} predictions)")


if __name__ == '__main__':
    main()