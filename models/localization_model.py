"""
第四章：基于对抗学习与嵌入对齐机制的智能合约漏洞子图生成与定位
包含：
  - MaskGenerator     : 边级掩码生成网络
  - SemanticAnchor    : 语义锚点集合（K-means构建）
  - VulnLocalizor     : 完整定位框架（对抗 + 嵌入对齐）
  - JointLoss         : 联合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data


# ═══════════════════════════════════════════════════════════
# 1. 边级掩码生成网络
# ═══════════════════════════════════════════════════════════

class MaskGenerator(nn.Module):
    """
    输入：边端点特征拼接 z_ij = h_i || h_j  (dim = 2d)
    输出：边重要性概率 m_ij ∈ (0,1)

    参数:
      node_dim : 节点嵌入维度 d
      hidden   : MLP 隐藏维度
    """
    def __init__(self, node_dim=256, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, node_emb, edge_index):
        """
        node_emb  : (N, d) 节点嵌入
        edge_index: (2, E) 边索引
        returns   : edge_mask (E,) 每条边的重要性概率
        """
        src, dst = edge_index
        z = torch.cat([node_emb[src], node_emb[dst]], dim=-1)  # (E, 2d)
        m = torch.sigmoid(self.mlp(z)).squeeze(-1)              # (E,)
        return m

    def apply_mask(self, edge_index, edge_attr, mask, threshold=None):
        """
        构造加权子图；threshold 不为 None 时做硬截断（推理阶段用）
        """
        if threshold is not None:
            keep = mask >= threshold
            return edge_index[:, keep], (edge_attr[keep] if edge_attr is not None else None), mask[keep]
        # 训练阶段：软加权（保留所有边，但权重不同）
        if edge_attr is not None:
            weighted_attr = edge_attr * mask.unsqueeze(-1)
        else:
            weighted_attr = mask.unsqueeze(-1)
        return edge_index, weighted_attr, mask

    def complement_mask(self, mask):
        return 1.0 - mask


# ═══════════════════════════════════════════════════════════
# 2. 语义锚点集合
# ═══════════════════════════════════════════════════════════

class SemanticAnchor:
    """
    基于 K-means 聚类构建语义锚点集合。
    使用冻结检测模型的中间表示（CFG 图级嵌入）进行聚类。
    """
    def __init__(self, n_clusters=6, device="cpu"):
        self.n_clusters = n_clusters
        self.device = device
        self.anchors = None   # Tensor (K, d)

    # ----------------------------------------------------------
    @torch.no_grad()
    def build(self, frozen_detector, train_loader):
        """
        从训练集漏洞样本中抽取 CFG 嵌入，执行 K-means，保存锚点。

        frozen_detector : MultiModalDetector（参数已冻结）
        train_loader    : 定位任务 DataLoader
        """
        frozen_detector.eval()
        embeddings = []
        for batch in train_loader:
            batch = batch.to(self.device)
            if batch.y.sum() == 0:
                continue
            # 仅对漏洞样本提取嵌入
            vuln_mask = batch.y == 1
            # 提取 CFG 图级嵌入
            node_emb = frozen_detector.cfg_encoder(
                batch.x, batch.edge_index, batch.batch)     # (B, d)
            embeddings.append(node_emb[vuln_mask].cpu().numpy())

        if len(embeddings) == 0:
            raise RuntimeError("没有找到漏洞样本，无法构建锚点")

        all_emb = np.concatenate(embeddings, axis=0)
        km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        km.fit(all_emb)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        self.anchors = centers.to(self.device)

    def anchor_distance_stat(self, graph_emb):
        """
        计算图嵌入到所有锚点的距离之和。
        graph_emb: (B, d)
        returns  : (B,) 标量统计量 s_G
        """
        # (B, 1, d) - (1, K, d) -> (B, K, d)
        diff = graph_emb.unsqueeze(1) - self.anchors.unsqueeze(0)
        dist = diff.norm(dim=-1)            # (B, K)
        return dist.sum(dim=-1)             # (B,)

    def alignment_loss(self, orig_stat, sub_stat):
        """
        L_align = || s_{G'} - s_G ||_2^2
        orig_stat, sub_stat: (B,)
        """
        return F.mse_loss(sub_stat, orig_stat.detach())


# ═══════════════════════════════════════════════════════════
# 3. 联合损失函数
# ═══════════════════════════════════════════════════════════

class JointLoss(nn.Module):
    """
    L = (1 - β - λ) * L_pred  +  β * L_uni  +  λ * L_align
    """
    def __init__(self, beta=0.3, lam=0.2, num_classes=2):
        super().__init__()
        assert beta + lam <= 1.0, "beta + lambda 必须 <= 1"
        self.beta = beta
        self.lam  = lam
        self.alpha = 1.0 - beta - lam   # 主任务权重
        self.uniform = torch.ones(num_classes) / num_classes

    def forward(self, logits_sub, logits_comp, labels, align_loss):
        """
        logits_sub  : (B, C) 漏洞子图预测 logits
        logits_comp : (B, C) 补集子图预测 logits
        labels      : (B,)   真实标签
        align_loss  : scalar 嵌入对齐损失
        """
        # 判别充分性：子图应维持漏洞预测
        L_pred = F.cross_entropy(logits_sub, labels)

        # 判别无关性：补集预测分布趋近均匀
        prob_comp = F.softmax(logits_comp, dim=-1)
        uniform = self.uniform.to(prob_comp.device)
        L_uni = F.kl_div(
            torch.log(prob_comp + 1e-8),
            uniform.expand_as(prob_comp),
            reduction="batchmean",
        )

        total = self.alpha * L_pred + self.beta * L_uni + self.lam * align_loss
        return total, {"L_pred": L_pred.item(),
                       "L_uni":  L_uni.item(),
                       "L_align": align_loss.item()}


# ═══════════════════════════════════════════════════════════
# 4. 完整漏洞定位框架
# ═══════════════════════════════════════════════════════════

class VulnLocalizor(nn.Module):
    """
    漏洞定位框架（第四章）
    - frozen_detector : 参数冻结的检测模型（MultiModalDetector）
    - mask_gen        : 掩码生成网络
    - anchor          : 语义锚点集合
    - criterion       : 联合损失函数
    """
    def __init__(self, frozen_detector, node_dim=256,
                 mask_hidden=128, beta=0.3, lam=0.2,
                 n_clusters=6, num_classes=2):
        super().__init__()
        # 检测模型参数冻结
        self.detector = frozen_detector
        for p in self.detector.parameters():
            p.requires_grad_(False)

        self.mask_gen = MaskGenerator(node_dim, mask_hidden)
        self.anchor   = SemanticAnchor(n_clusters)
        self.criterion= JointLoss(beta, lam, num_classes)
        self.node_dim = node_dim

    # ----------------------------------------------------------
    def _get_node_emb(self, data):
        """从冻结的 CFGEncoder 中取节点级嵌入（不做 readout）"""
        encoder = self.detector.cfg_encoder
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        for conv in encoder.convs:
            x = conv(x, edge_index, num_nodes)
            x = F.dropout(x, p=0.0, training=False)   # 推理阶段不 dropout
        return x   # (N, d)

    # ----------------------------------------------------------
    def _forward_with_mask(self, data, mask, batch):
        """
        用加权邻接矩阵经过 CFGEncoder 的 DGCNConv 层做前向传播，
        再经检测模型 classifier 输出 logits。

        mask: (E,) 边权重
        """
        # 将边权重乘到边特征上（软掩码）
        encoder = self.detector.cfg_encoder
        x_init = data.x
        edge_index = data.edge_index
        num_nodes = x_init.size(0)

        x = x_init
        for conv in encoder.convs:
            # 手动实现带 mask 的有向聚合
            src, dst = edge_index
            # 出边聚合
            agg_in = torch.zeros_like(x)
            weighted_msg = x[src] * mask.unsqueeze(-1)
            agg_in.scatter_add_(0, dst.unsqueeze(1).expand_as(weighted_msg),
                                weighted_msg)
            # 入边聚合
            agg_out = torch.zeros_like(x)
            weighted_msg2 = x[dst] * mask.unsqueeze(-1)
            agg_out.scatter_add_(0, src.unsqueeze(1).expand_as(weighted_msg2),
                                 weighted_msg2)
            x = conv.W_self(x) + conv.W_in(agg_in) + conv.W_out(agg_out)
            x = conv.bn(F.relu(x))

        graph_emb = global_mean_pool(x, batch)  # (B, d)
        logits = self.detector.classifier(graph_emb)
        return logits, graph_emb

    # ----------------------------------------------------------
    def forward(self, data):
        """
        训练阶段前向计算
        返回: total_loss, loss_dict, edge_mask
        """
        batch = data.batch
        labels = data.y

        # 1) 节点嵌入（冻结）
        node_emb = self._get_node_emb(data)

        # 2) 掩码生成
        mask = self.mask_gen(node_emb, data.edge_index)      # (E,)
        comp_mask = self.mask_gen.complement_mask(mask)       # (E,)

        # 3) 子图 & 补集图前向
        logits_sub,  emb_sub  = self._forward_with_mask(data, mask,      batch)
        logits_comp, _        = self._forward_with_mask(data, comp_mask, batch)

        # 4) 原始图嵌入（用于对齐）
        with torch.no_grad():
            _, orig_emb = self._forward_with_mask(
                data, torch.ones_like(mask), batch)

        # 5) 锚点距离统计
        s_orig = self.anchor.anchor_distance_stat(orig_emb)   # (B,)
        s_sub  = self.anchor.anchor_distance_stat(emb_sub)    # (B,)
        align_loss = self.anchor.alignment_loss(s_orig, s_sub)

        # 6) 联合损失
        total_loss, loss_dict = self.criterion(
            logits_sub, logits_comp, labels, align_loss)

        return total_loss, loss_dict, mask

    # ----------------------------------------------------------
    @torch.no_grad()
    def predict_node_scores(self, data):
        """
        推理阶段：返回每个节点的漏洞风险得分
        得分 = 以该节点为端点的边掩码均值
        """
        node_emb = self._get_node_emb(data)
        mask = self.mask_gen(node_emb, data.edge_index)   # (E,)

        src, dst = data.edge_index
        num_nodes = data.x.size(0)

        # 对每个节点，聚合其相关边的掩码均值
        node_score = torch.zeros(num_nodes, device=mask.device)
        count = torch.zeros(num_nodes, device=mask.device)

        node_score.scatter_add_(0, src, mask)
        node_score.scatter_add_(0, dst, mask)
        count.scatter_add_(0, src, torch.ones_like(mask))
        count.scatter_add_(0, dst, torch.ones_like(mask))
        count = count.clamp(min=1)
        node_score = node_score / count

        return node_score   # (N,)