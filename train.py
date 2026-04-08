"""
训练脚本
  - train_detection   : 第三章多模态漏洞检测模型训练
  - train_localization: 第四章漏洞定位模型训练
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from models.detection_model import MultiModalDetector
from models.localization_model import VulnLocalizor
from data.dataset import get_detection_loaders, get_localization_loaders


# ═══════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt["epoch"]


# ═══════════════════════════════════════════════════════════
# 第三章 检测模型训练
# ═══════════════════════════════════════════════════════════

def train_detection(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, val_loader, test_loader = get_detection_loaders(
        args.data_root, vuln_type=args.vuln_type,
        batch_size=args.batch_size, seed=args.seed)

    # 模型
    model = MultiModalDetector(
        cpg_in_dim=args.cpg_in_dim,
        cpg_hidden=args.hidden_dim,
        cpg_out=args.hidden_dim,
        bytecode_out=args.hidden_dim,
        ngram_in=args.ngram_dim,
        ngram_out=args.hidden_dim,
        cfg_in_dim=args.cfg_in_dim,
        cfg_hidden=args.hidden_dim,
        cfg_out=args.hidden_dim,
        unified_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for cpg, cfg, img, ngram, labels in train_loader:
            cpg   = cpg.to(device)
            cfg   = cfg.to(device)
            img   = img.to(device)
            ngram = ngram.to(device)
            labels= labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(cpg, cfg, img, ngram)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ---- validate ----
        val_metrics = evaluate_detection(model, val_loader, device)
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"acc={val_metrics['acc']:.4f} | "
              f"f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(args.ckpt_dir, "detection_best.pt"))

    # ---- test ----
    load_checkpoint(model, None,
                    os.path.join(args.ckpt_dir, "detection_best.pt"), device)
    test_metrics = evaluate_detection(model, test_loader, device)
    print("\n===== Test Results =====")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    return model


@torch.no_grad()
def evaluate_detection(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for cpg, cfg, img, ngram, labels in loader:
        cpg   = cpg.to(device)
        cfg   = cfg.to(device)
        img   = img.to(device)
        ngram = ngram.to(device)
        logits, _ = model(cpg, cfg, img, ngram)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ═══════════════════════════════════════════════════════════
# 第四章 定位模型训练
# ═══════════════════════════════════════════════════════════

def train_localization(args, frozen_detector):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_localization_loaders(
        args.loc_data_root, batch_size=args.batch_size, seed=args.seed)

    model = VulnLocalizor(
        frozen_detector=frozen_detector.to(device),
        node_dim=args.hidden_dim,
        mask_hidden=128,
        beta=args.beta,
        lam=args.lam,
        n_clusters=args.n_clusters,
    ).to(device)

    # 构建语义锚点（仅训练集漏洞样本）
    print("构建语义锚点...")
    model.anchor.device = device
    model.anchor.build(frozen_detector, train_loader)

    # 只优化掩码生成网络参数
    optimizer = Adam(model.mask_gen.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auroc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, loss_dict, _ = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        val_auroc = evaluate_localization_auroc(model, val_loader, device)
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"val_AUROC={val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            save_checkpoint(model.mask_gen, optimizer, epoch,
                            os.path.join(args.ckpt_dir, "localization_best.pt"))

    # ---- test ----
    load_checkpoint(model.mask_gen, None,
                    os.path.join(args.ckpt_dir, "localization_best.pt"), device)
    test_metrics = evaluate_localization_full(model, test_loader, device)
    print("\n===== Test Localization Results =====")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    return model


@torch.no_grad()
def evaluate_localization_auroc(model, loader, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        scores = model.predict_node_scores(batch)
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(batch.node_label.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    if all_labels.sum() == 0:
        return 0.0
    return roc_auc_score(all_labels, all_scores)


@torch.no_grad()
def evaluate_localization_full(model, loader, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        scores = model.predict_node_scores(batch)
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(batch.node_label.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    auroc = roc_auc_score(all_labels, all_scores) if all_labels.sum() > 0 else 0.0
    topn = compute_topn(all_scores, all_labels, [1, 3, 5, 10, 20])
    results = {"AUROC": auroc}
    results.update(topn)
    return results


def compute_topn(scores, labels, ns):
    """
    Top-N 准确率：在每个样本排名前 N 节点中是否命中漏洞节点。
    简化实现：将全部节点统一排序后计算全局 Top-N 覆盖率。
    """
    sorted_idx = np.argsort(-scores)
    results = {}
    total_vuln = labels.sum()
    for n in ns:
        top_n_idx = sorted_idx[:n]
        hit = labels[top_n_idx].sum()
        results[f"Top-{n}"] = float(hit) / max(total_vuln, 1)
    return results


# ═══════════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    # 通用
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--ckpt_dir",   type=str,   default="checkpoints")
    # 检测
    parser.add_argument("--data_root",  type=str,   default="data/detection")
    parser.add_argument("--vuln_type",  type=str,   default="reentrancy")
    parser.add_argument("--cpg_in_dim", type=int,   default=128)
    parser.add_argument("--cfg_in_dim", type=int,   default=128)
    parser.add_argument("--ngram_dim",  type=int,   default=5000)
    # 定位
    parser.add_argument("--loc_data_root", type=str, default="data/localization")
    parser.add_argument("--beta",       type=float, default=0.3)
    parser.add_argument("--lam",        type=float, default=0.2)
    parser.add_argument("--n_clusters", type=int,   default=6)
    parser.add_argument("--task",       type=str,   default="detection",
                        choices=["detection", "localization", "all"])
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    if args.task in ("detection", "all"):
        print("=" * 50)
        print("第三章：多模态漏洞检测模型训练")
        print("=" * 50)
        detector = train_detection(args)

    if args.task in ("localization", "all"):
        print("\n" + "=" * 50)
        print("第四章：漏洞定位模型训练")
        print("=" * 50)
        if args.task == "localization":
            # 单独训练定位模块时需先加载检测模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            detector = MultiModalDetector(
                cpg_in_dim=args.cpg_in_dim, cpg_hidden=args.hidden_dim,
                cpg_out=args.hidden_dim,   bytecode_out=args.hidden_dim,
                ngram_in=args.ngram_dim,   ngram_out=args.hidden_dim,
                cfg_in_dim=args.cfg_in_dim, cfg_hidden=args.hidden_dim,
                cfg_out=args.hidden_dim,   unified_dim=args.hidden_dim,
            ).to(device)
            ckpt_path = os.path.join(args.ckpt_dir, "detection_best.pt")
            if os.path.exists(ckpt_path):
                load_checkpoint(detector, None, ckpt_path, device)
                print(f"已加载检测模型：{ckpt_path}")
            else:
                print("警告：未找到检测模型权重，使用随机初始化")

        train_localization(args, detector)