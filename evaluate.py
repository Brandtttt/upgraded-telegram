"""
评估脚本
  - 第三章：消融实验、注意力权重分析、与基线对比
  - 第四章：Top-N准确率、AUROC、保真度分析、消融实验、时间分析
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "SimHei"  # 支持中文
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from collections import defaultdict

from models.detection_model import MultiModalDetector
from models.localization_model import VulnLocalizor
from train import (evaluate_detection, evaluate_localization_full,
                   compute_topn, set_seed,
                   get_detection_loaders, get_localization_loaders)


# ═══════════════════════════════════════════════════════════
# 第三章实验
# ═══════════════════════════════════════════════════════════

# ----------------------------------------------------------
# 3.1 超参数搜索：DGCN×RGCN 层数热力图
# ----------------------------------------------------------

def exp_layer_heatmap(args):
    """
    遍历 DGCN 和 RGCN 层数组合 (2~5)，记录 F1 分数，绘制热力图。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_detection_loaders(
        args.data_root, args.vuln_type,
        batch_size=args.batch_size, seed=args.seed)

    layer_range = [2, 3, 4, 5]
    f1_matrix = np.zeros((len(layer_range), len(layer_range)))

    for i, n_dgcn in enumerate(layer_range):
        for j, n_rgcn in enumerate(layer_range):
            model = MultiModalDetector(
                cpg_in_dim=args.cpg_in_dim,
                cpg_hidden=args.hidden_dim, cpg_out=args.hidden_dim,
                bytecode_out=args.hidden_dim,
                ngram_in=args.ngram_dim, ngram_out=args.hidden_dim,
                cfg_in_dim=args.cfg_in_dim,
                cfg_hidden=args.hidden_dim, cfg_out=args.hidden_dim,
                unified_dim=args.hidden_dim,
            ).to(device)

            # 覆盖层数
            from models.detection_model import CPGEncoder, CFGEncoder
            model.cpg_encoder = CPGEncoder(
                args.cpg_in_dim, args.hidden_dim, args.hidden_dim,
                num_layers=n_rgcn).to(device)
            model.cfg_encoder = CFGEncoder(
                args.cfg_in_dim, args.hidden_dim, args.hidden_dim,
                num_layers=n_dgcn).to(device)

            _quick_train(model, train_loader, device, epochs=args.epochs, lr=0.001)
            metrics = evaluate_detection(model, val_loader, device)
            f1_matrix[i][j] = metrics["f1"] * 100
            print(f"DGCN={n_dgcn}, RGCN={n_rgcn} → F1={f1_matrix[i][j]:.2f}%")

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(f1_matrix, cmap="YlOrRd", vmin=f1_matrix.min()-1,
                   vmax=f1_matrix.max()+1)
    ax.set_xticks(range(len(layer_range)))
    ax.set_yticks(range(len(layer_range)))
    ax.set_xticklabels([f"RGCN={l}" for l in layer_range])
    ax.set_yticklabels([f"DGCN={l}" for l in layer_range])
    for i in range(len(layer_range)):
        for j in range(len(layer_range)):
            ax.text(j, i, f"{f1_matrix[i,j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="black" if f1_matrix[i,j] < f1_matrix.max()-2 else "white")
    plt.colorbar(im, ax=ax, label="F1 (%)")
    ax.set_title("不同 DGCN 层数与 RGCN 层数组合下模型的 F1 分数（%）")
    plt.tight_layout()
    plt.savefig("results/heatmap1.png", dpi=150)
    plt.close()
    return f1_matrix


# ----------------------------------------------------------
# 3.2 学习率敏感性实验（检测模型）
# ----------------------------------------------------------

def exp_lr_detection(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_detection_loaders(
        args.data_root, args.vuln_type,
        batch_size=args.batch_size, seed=args.seed)

    lrs = [0.0001, 0.001, 0.01, 0.1]
    f1_scores = []

    for lr in lrs:
        model = MultiModalDetector(
            cpg_in_dim=args.cpg_in_dim, cpg_hidden=args.hidden_dim,
            cpg_out=args.hidden_dim,    bytecode_out=args.hidden_dim,
            ngram_in=args.ngram_dim,   ngram_out=args.hidden_dim,
            cfg_in_dim=args.cfg_in_dim, cfg_hidden=args.hidden_dim,
            cfg_out=args.hidden_dim,   unified_dim=args.hidden_dim,
        ).to(device)
        _quick_train(model, train_loader, device, epochs=args.epochs, lr=lr)
        metrics = evaluate_detection(model, val_loader, device)
        f1_scores.append(metrics["f1"] * 100)
        print(f"LR={lr} → F1={metrics['f1']*100:.2f}%")

    # 柱状图
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([str(l) for l in lrs], f1_scores,
                  color=["#5B9BD5", "#ED7D31", "#A9D18E", "#FF0000"])
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(min(f1_scores)-5, 100)
    ax.set_xlabel("学习率")
    ax.set_ylabel("F1 分数 (%)")
    ax.set_title("不同学习率下模型的 F1 分数对比")
    plt.tight_layout()
    plt.savefig("results/f1_lr.png", dpi=150)
    plt.close()
    return dict(zip(lrs, f1_scores))


# ----------------------------------------------------------
# 3.3 消融实验（检测模型）
# ----------------------------------------------------------

def exp_ablation_detection(args, full_model, test_loader):
    """
    依次移除各模态 / 注意力机制，评估 F1 下降。
    full_model: 已训练好的完整模型
    """
    device = next(full_model.parameters()).device
    results = {}

    # 完整模型
    metrics = evaluate_detection(full_model, test_loader, device)
    results["本方法"] = metrics

    # 禁用各模态：将对应 encoder 输出置零
    ablation_cases = {
        "w/o 源码":     "cpg",
        "w/o 字节码":   "bytecode",
        "w/o n-gram":   "ngram",
        "w/o CFG":      "cfg",
        "w/o 注意力机制": "attn",
    }
    for label, ablate in ablation_cases.items():
        metrics = evaluate_detection_ablation(
            full_model, test_loader, device, ablate_modal=ablate)
        results[label] = metrics
        print(f"{label}: F1={metrics['f1']*100:.2f}%")

    _print_detection_table(results)
    return results


@torch.no_grad()
def evaluate_detection_ablation(model, loader, device, ablate_modal="cpg"):
    model.eval()
    all_preds, all_labels = [], []
    for cpg, cfg, img, ngram, labels in loader:
        cpg   = cpg.to(device)
        cfg   = cfg.to(device)
        img   = img.to(device)
        ngram = ngram.to(device)

        # 提取各模态特征
        f_cpg   = model.cpg_encoder(cpg.x, cpg.edge_index, cpg.edge_type, cpg.batch)
        f_byte  = model.bytecode_encoder(img)
        f_ngram = model.ngram_encoder(ngram)
        f_cfg   = model.cfg_encoder(cfg.x, cfg.edge_index, cfg.batch)

        # 对应模态置零
        zero_map = {
            "cpg":     lambda: f_cpg.zero_(),
            "bytecode":lambda: f_byte.zero_(),
            "ngram":   lambda: f_ngram.zero_(),
            "cfg":     lambda: f_cfg.zero_(),
        }
        if ablate_modal in zero_map:
            zero_map[ablate_modal]()

        if ablate_modal == "attn":
            # 替换为简单均值融合
            modal_features = {
                "cpg": f_cpg, "bytecode": f_byte,
                "ngram": f_ngram, "cfg": f_cfg}
            unified = sum(model.fusion.proj[k](
                          torch.sigmoid(model.fusion.self_attn[k](v)) * v)
                          for k, v in modal_features.items()) / 4
            fused = unified
        else:
            fused, _ = model.fusion({
                "cpg": f_cpg, "bytecode": f_byte,
                "ngram": f_ngram, "cfg": f_cfg})

        logits = model.classifier(fused)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def _print_detection_table(results):
    print("\n" + "=" * 70)
    print(f"{'方法':<20} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'F1':>8}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<20} {m['acc']*100:>7.2f}% {m['prec']*100:>7.2f}% "
              f"{m['rec']*100:>7.2f}% {m['f1']*100:>7.2f}%")
    print("=" * 70)


# ----------------------------------------------------------
# 3.4 模态注意力权重分析（按漏洞类型）
# ----------------------------------------------------------

@torch.no_grad()
def analyze_attention_weights(model, loader, device, vuln_types):
    """
    统计不同漏洞类型下各模态的平均注意力权重。
    vuln_types: list[str], 与 label 对应
    """
    model.eval()
    weight_accum = defaultdict(lambda: defaultdict(list))

    for cpg, cfg, img, ngram, labels in loader:
        cpg   = cpg.to(device)
        cfg   = cfg.to(device)
        img   = img.to(device)
        ngram = ngram.to(device)
        _, weights = model(cpg, cfg, img, ngram)

        for i, lbl in enumerate(labels.numpy()):
            vt = vuln_types[lbl]
            for modal, w in weights.items():
                weight_accum[vt][modal].append(w[i].item())

    print("\n===== 各漏洞类型下模态注意力权重 =====")
    for vt, modal_dict in weight_accum.items():
        avg = {m: np.mean(v) for m, v in modal_dict.items()}
        print(f"  {vt}: " +
              ", ".join(f"{m}={v:.2f}" for m, v in avg.items()))
    return weight_accum


# ═══════════════════════════════════════════════════════════
# 第四章实验
# ═══════════════════════════════════════════════════════════

# ----------------------------------------------------------
# 4.1 锚点数量实验
# ----------------------------------------------------------

def exp_anchor_num(args, frozen_detector, train_loader, val_loader):
    device = next(frozen_detector.parameters()).device
    anchor_ns = [3, 4, 5, 6, 7]
    aurocs = []
    for k in anchor_ns:
        model = VulnLocalizor(
            frozen_detector, node_dim=args.hidden_dim,
            beta=args.beta, lam=args.lam, n_clusters=k).to(device)
        model.anchor.device = device
        model.anchor.build(frozen_detector, train_loader)
        _quick_train_localization(model, train_loader, device, epochs=args.epochs)
        from train import evaluate_localization_auroc
        auroc = evaluate_localization_auroc(model, val_loader, device)
        aurocs.append(auroc)
        print(f"K={k} → AUROC={auroc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(anchor_ns, aurocs, marker="o", color="#5B9BD5", linewidth=2)
    for x, y in zip(anchor_ns, aurocs):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("锚点数量 K")
    ax.set_ylabel("AUROC")
    ax.set_title("锚点数对 AUROC 的影响")
    ax.set_ylim(0.75, 0.95)
    plt.tight_layout()
    plt.savefig("results/anchor_para.png", dpi=150)
    plt.close()
    return dict(zip(anchor_ns, aurocs))


# ----------------------------------------------------------
# 4.2 权重系数热力图 (λ, β)
# ----------------------------------------------------------

def exp_weight_heatmap(args, frozen_detector, train_loader, val_loader):
    device = next(frozen_detector.parameters()).device
    betas   = np.arange(0.0, 1.0, 0.1)
    lambdas = np.arange(0.0, 1.0, 0.1)
    auroc_mat = np.zeros((len(betas), len(lambdas)))

    from train import evaluate_localization_auroc
    for i, beta in enumerate(betas):
        for j, lam in enumerate(lambdas):
            if beta + lam > 1.0:
                auroc_mat[i, j] = np.nan
                continue
            model = VulnLocalizor(
                frozen_detector, node_dim=args.hidden_dim,
                beta=float(beta), lam=float(lam),
                n_clusters=args.n_clusters).to(device)
            model.anchor.device = device
            model.anchor.build(frozen_detector, train_loader)
            _quick_train_localization(model, train_loader, device, epochs=args.epochs)
            auroc = evaluate_localization_auroc(model, val_loader, device)
            auroc_mat[i, j] = auroc

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(auroc_mat, cmap="YlOrRd", vmin=0.5, vmax=0.92)
    ax.set_xticks(range(len(lambdas)))
    ax.set_yticks(range(len(betas)))
    ax.set_xticklabels([f"{l:.1f}" for l in lambdas], fontsize=8)
    ax.set_yticklabels([f"{b:.1f}" for b in betas],   fontsize=8)
    ax.set_xlabel("λ（嵌入对齐权重）")
    ax.set_ylabel("β（对抗约束权重）")
    ax.set_title("不同 (λ, β) 组合下的 AUROC 热力图")
    plt.colorbar(im, ax=ax, label="AUROC")
    plt.tight_layout()
    plt.savefig("results/heatmap.png", dpi=150)
    plt.close()
    return auroc_mat


# ----------------------------------------------------------
# 4.3 保真度分析
# ----------------------------------------------------------

@torch.no_grad()
def fidelity_analysis(model, test_loader, device, max_del=14):
    """
    逐步删除最重要的边，记录预测概率变化。
    返回: steps (list[int]), probs (list[float])
    """
    model.eval()
    all_prob_curves = []

    for batch in test_loader:
        batch = batch.to(device)
        vuln_mask = batch.y == 1
        if vuln_mask.sum() == 0:
            continue

        # 只处理漏洞样本中的第一个（简化）
        first_idx = vuln_mask.nonzero()[0, 0].item()

        node_emb = model._get_node_emb(batch)
        edge_mask = model.mask_gen(node_emb, batch.edge_index)

        # 按重要性排序边
        sorted_edges = edge_mask.argsort(descending=True).cpu().numpy()

        prob_curve = []
        current_mask = torch.ones_like(edge_mask)

        # 0条删除时的初始概率
        logits, _ = model._forward_with_mask(batch, current_mask, batch.batch)
        prob = F.softmax(logits, dim=-1)[first_idx, 1].item()
        prob_curve.append(prob)

        for step in range(min(max_del, len(sorted_edges))):
            current_mask[sorted_edges[step]] = 0.0
            logits, _ = model._forward_with_mask(batch, current_mask, batch.batch)
            prob = F.softmax(logits, dim=-1)[first_idx, 1].item()
            prob_curve.append(prob)

        all_prob_curves.append(prob_curve)
        if len(all_prob_curves) >= 100:
            break

    # 对齐长度
    min_len = min(len(c) for c in all_prob_curves)
    avg_curve = np.mean([c[:min_len] for c in all_prob_curves], axis=0)
    return list(range(min_len)), avg_curve.tolist()


def plot_fidelity(results_dict, save_path="results/fidelity.png"):
    """
    results_dict: {method_name: (steps, probs)}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"本方法": "#E74C3C", "ReVulDL": "#3498DB", "PGExplainer": "#2ECC71"}
    for name, (steps, probs) in results_dict.items():
        ax.plot(steps, probs, marker="o", label=name,
                color=colors.get(name, "#888888"), linewidth=2)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.6, label="随机猜测水平")
    ax.set_xlabel("删除边数量")
    ax.set_ylabel("漏洞预测概率")
    ax.set_title("保真度分析")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------------------------------------
# 4.4 Top-N 对比实验
# ----------------------------------------------------------

def plot_topn_table(results_dict):
    """
    results_dict: {method: {Top-1: ..., Top-3: ..., ...}}
    """
    print("\n" + "=" * 70)
    print(f"{'方法':<16} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} "
          f"{'Top-10':>8} {'Top-20':>8}")
    print("-" * 70)
    for name, topn in results_dict.items():
        vals = [topn.get(f"Top-{n}", 0) * 100
                for n in [1, 3, 5, 10, 20]]
        print(f"{name:<16}" + "".join(f"{v:>7.2f}%" for v in vals))
    print("=" * 70)


# ----------------------------------------------------------
# 4.5 消融实验（定位模型）
# ----------------------------------------------------------

def exp_ablation_localization(args, frozen_detector,
                               train_loader, val_loader, test_loader):
    device = next(frozen_detector.parameters()).device
    from train import evaluate_localization_auroc

    configs = {
        "本方法 (adv+anc)":      (True,  True),
        "w/o 嵌入对齐 (adv only)": (True,  False),
        "w/o 对抗学习 (anc only)": (False, True),
        "baseline (无两模块)":    (False, False),
    }
    aurocs = {}
    for name, (use_adv, use_anc) in configs.items():
        lam  = args.lam  if use_anc else 0.0
        beta = args.beta if use_adv else 0.0
        model = VulnLocalizor(
            frozen_detector, node_dim=args.hidden_dim,
            beta=beta, lam=lam, n_clusters=args.n_clusters).to(device)
        model.anchor.device = device
        if use_anc:
            model.anchor.build(frozen_detector, train_loader)
        else:
            # 无锚点：置全零锚点（对齐损失无效）
            model.anchor.anchors = torch.zeros(
                args.n_clusters, args.hidden_dim, device=device)

        _quick_train_localization(model, train_loader, device, epochs=args.epochs)
        auroc = evaluate_localization_auroc(model, val_loader, device)
        aurocs[name] = auroc
        print(f"{name}: AUROC={auroc:.4f}")

    # 柱状图
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(list(aurocs.keys()), list(aurocs.values()),
                  color=["#E74C3C", "#F39C12", "#3498DB", "#95A5A6"])
    for bar, val in zip(bars, aurocs.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("AUROC")
    ax.set_title("消融实验")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("results/ablation1.png", dpi=150)
    plt.close()
    return aurocs


# ----------------------------------------------------------
# 4.6 时间复杂度分析
# ----------------------------------------------------------

def time_analysis(models_dict, test_loader, device, n_contracts=100):
    """
    models_dict: {method_name: callable(batch)->scores}
    """
    times = {}
    batches = []
    cnt = 0
    for batch in test_loader:
        batches.append(batch.to(device))
        cnt += batch.num_graphs
        if cnt >= n_contracts:
            break

    for name, fn in models_dict.items():
        t0 = time.perf_counter()
        for batch in batches:
            with torch.no_grad():
                fn(batch)
        elapsed = (time.perf_counter() - t0) / cnt * 1000  # ms/contract
        times[name] = elapsed
        print(f"{name}: {elapsed:.1f} ms / contract")

    # 柱状图
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(list(times.keys()), list(times.values()),
                  color=["#E74C3C","#F39C12","#3498DB","#2ECC71","#9B59B6"])
    for bar, val in zip(bars, times.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("平均耗时 (ms / 合约)")
    ax.set_title("时间复杂度分析（100 个合约）")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("results/time.png", dpi=150)
    plt.close()
    return times


# ═══════════════════════════════════════════════════════════
# 内部工具函数
# ═══════════════════════════════════════════════════════════

def _quick_train(model, train_loader, device, epochs=50, lr=0.001):
    """快速训练（用于超参数搜索）"""
    import torch.nn as nn
    from torch.optim import SGD
    model.train()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for cpg, cfg, img, ngram, labels in train_loader:
            cpg   = cpg.to(device)
            cfg   = cfg.to(device)
            img   = img.to(device)
            ngram = ngram.to(device)
            labels= labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(cpg, cfg, img, ngram)
            criterion(logits, labels).backward()
            optimizer.step()


def _quick_train_localization(model, train_loader, device, epochs=50):
    from torch.optim import Adam
    optimizer = Adam(model.mask_gen.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(batch)
            loss.backward()
            optimizer.step()