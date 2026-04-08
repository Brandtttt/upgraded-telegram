"""
第三章：基于多模态特征融合与注意力机制的智能合约漏洞检测模型
包含：
  - CPGEncoder        : RGCN 编码 CPG 图
  - BytecodeGrayNet   : CNN + SPP 编码字节码灰度图
  - CFGEncoder        : DGCN 编码操作码 CFG 图
  - AttentionFusion   : 两级注意力融合
  - MultiModalDetector: 完整检测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj


# ═══════════════════════════════════════════════════════════
# 1. CPG 编码器 (RGCN)
# ═══════════════════════════════════════════════════════════

class CPGEncoder(nn.Module):
    """
    关系图卷积网络，编码代码属性图
    支持三类边：syntax(0), control(1), dataflow(2)
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_relations=3, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(dims[i], dims[i + 1], num_relations=num_relations)
            )
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        # 图级读出
        out = global_mean_pool(x, batch)   # (B, out_dim)
        return out


# ═══════════════════════════════════════════════════════════
# 2. 字节码灰度图编码器 (Conv + ResBlock + SPP)
# ═══════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class SPPLayer(nn.Module):
    """空间金字塔池化：输出固定长度特征向量"""
    def __init__(self, scales=(1, 2, 3, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, x):
        # x: (B, C, H, W)
        parts = []
        for s in self.scales:
            pooled = F.adaptive_max_pool2d(x, (s, s))  # (B, C, s, s)
            parts.append(pooled.view(x.size(0), -1))   # (B, C*s*s)
        return torch.cat(parts, dim=1)                  # (B, C * sum(s^2))


class BytecodeGrayNet(nn.Module):
    """
    输入: (B, 1, H, W) 灰度图
    输出: (B, 64 * 30)  = (B, 1920)
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.spp  = SPPLayer(scales=(1, 2, 3, 4))   # 1+4+9+16=30 bins
        spp_dim = 64 * (1 + 4 + 9 + 16)              # 1920
        self.proj = nn.Linear(spp_dim, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.spp(x)
        x = F.relu(self.proj(x))
        return x                   # (B, out_dim)


# ═══════════════════════════════════════════════════════════
# 3. 操作码 CFG 编码器 (有向 GCN)
# ═══════════════════════════════════════════════════════════

class DGCNConv(nn.Module):
    """
    简单有向 GCN：分别聚合入边与出边，再与自身拼接后线性变换。
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_in  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_out = nn.Linear(in_dim, out_dim, bias=False)
        self.W_self= nn.Linear(in_dim, out_dim, bias=False)
        self.bn    = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, num_nodes):
        src, dst = edge_index
        # 出边聚合（dst 收集 src 的信息）
        agg_in = torch.zeros(num_nodes, x.size(1), device=x.device)
        agg_in.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        # 入边聚合（src 收集 dst 的信息）
        agg_out = torch.zeros(num_nodes, x.size(1), device=x.device)
        agg_out.scatter_add_(0, src.unsqueeze(1).expand(-1, x.size(1)), x[dst])

        out = self.W_self(x) + self.W_in(agg_in) + self.W_out(agg_out)
        out = self.bn(F.relu(out))
        return out


class CFGEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(DGCNConv(dims[i], dims[i + 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        num_nodes = x.size(0)
        for conv in self.convs:
            x = conv(x, edge_index, num_nodes)
            x = self.dropout(x)
        return global_mean_pool(x, batch)  # (B, out_dim)


# ═══════════════════════════════════════════════════════════
# 4. 两级注意力融合模块
# ═══════════════════════════════════════════════════════════

class AttentionFusion(nn.Module):
    """
    Level-1: 模态内自注意力 (逐元素 sigmoid 门控)
    Level-2: 模态间 Softmax 权重
    """
    def __init__(self, modal_dims: dict, unified_dim=256):
        """
        modal_dims: {'cpg': d1, 'bytecode': d2, 'ngram': d3, 'cfg': d4}
        """
        super().__init__()
        self.modals = list(modal_dims.keys())

        # 模态内自注意力（各自独立）
        self.self_attn = nn.ModuleDict({
            k: nn.Linear(d, d) for k, d in modal_dims.items()
        })
        # 投影到统一维度
        self.proj = nn.ModuleDict({
            k: nn.Linear(d, unified_dim) for k, d in modal_dims.items()
        })
        # 模态间 MLP（输入为所有模态拼接）
        total_dim = unified_dim * len(modal_dims)
        self.inter_attn = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(modal_dims)),
        )

    def forward(self, modal_features: dict):
        """
        modal_features: {modal_name: tensor (B, d)}
        returns: fused (B, unified_dim), weights dict
        """
        refined = {}
        for k, feat in modal_features.items():
            gate = torch.sigmoid(self.self_attn[k](feat))
            refined[k] = F.relu(self.proj[k](gate * feat))

        concat = torch.cat([refined[k] for k in self.modals], dim=1)
        scores = self.inter_attn(concat)          # (B, num_modals)
        weights = F.softmax(scores, dim=-1)       # (B, num_modals)

        fused = sum(
            weights[:, i:i+1] * refined[self.modals[i]]
            for i in range(len(self.modals))
        )
        weight_dict = {k: weights[:, i] for i, k in enumerate(self.modals)}
        return fused, weight_dict


# ═══════════════════════════════════════════════════════════
# 5. 完整多模态检测模型
# ═══════════════════════════════════════════════════════════

class MultiModalDetector(nn.Module):
    """
    完整多模态漏洞检测模型（第三章）
    输入: CPG图, 字节码灰度图, N-gram向量, CFG图
    输出: (logits, weight_dict)
    """
    def __init__(self,
                 cpg_in_dim=128, cpg_hidden=256, cpg_out=256,
                 bytecode_out=256,
                 ngram_in=5000, ngram_out=256,
                 cfg_in_dim=128, cfg_hidden=256, cfg_out=256,
                 unified_dim=256,
                 num_classes=2, dropout=0.2):
        super().__init__()

        self.cpg_encoder = CPGEncoder(
            cpg_in_dim, cpg_hidden, cpg_out, num_layers=3, dropout=dropout)
        self.bytecode_encoder = BytecodeGrayNet(out_dim=bytecode_out)
        self.ngram_encoder = nn.Sequential(
            nn.Linear(ngram_in, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, ngram_out),
            nn.ReLU(),
        )
        self.cfg_encoder = CFGEncoder(
            cfg_in_dim, cfg_hidden, cfg_out, num_layers=3, dropout=dropout)

        modal_dims = {
            "cpg":      cpg_out,
            "bytecode": bytecode_out,
            "ngram":    ngram_out,
            "cfg":      cfg_out,
        }
        self.fusion = AttentionFusion(modal_dims, unified_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(unified_dim, num_classes),
        )

    # ----------------------------------------------------------
    def forward(self, cpg_data, cfg_data, bytecode_img, ngram_vec):
        """
        cpg_data    : PyG Batch (x, edge_index, edge_type, batch)
        cfg_data    : PyG Batch (x, edge_index, batch)
        bytecode_img: (B, 1, H, W)
        ngram_vec   : (B, ngram_in)
        """
        f_cpg = self.cpg_encoder(
            cpg_data.x, cpg_data.edge_index,
            cpg_data.edge_type, cpg_data.batch)

        f_byte = self.bytecode_encoder(bytecode_img)

        f_ngram = self.ngram_encoder(ngram_vec)

        f_cfg = self.cfg_encoder(
            cfg_data.x, cfg_data.edge_index, cfg_data.batch)

        fused, weights = self.fusion({
            "cpg":      f_cpg,
            "bytecode": f_byte,
            "ngram":    f_ngram,
            "cfg":      f_cfg,
        })

        logits = self.classifier(fused)
        return logits, weights

    # ----------------------------------------------------------
    def get_graph_embedding(self, cfg_data):
        """供第四章定位模块调用，提取CFG的中间嵌入"""
        return self.cfg_encoder(
            cfg_data.x, cfg_data.edge_index, cfg_data.batch)