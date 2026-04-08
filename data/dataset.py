"""
数据集处理模块
支持SmartBugs-Wild 和 SolidFI-Benchmark 数据集
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split


# ────────────────────────────────────────────────────────────
# 第三章 检测任务数据集
# ────────────────────────────────────────────────────────────

class SmartContractDetectionDataset(Dataset):
    """
    多模态智能合约漏洞检测数据集
    每个样本包含:
      - cpg_graph  : CPG图 (torch_geometric.data.Data)
      - bytecode_img: 字节码灰度图 (1 x H x W tensor)
      - ngram_vec  : N-gram 操作码频率向量
      - cfg_graph  : 操作码CFG图 (torch_geometric.data.Data)
      - label      : 0/1 标签
    """

    def __init__(self, data_root, vuln_type="reentrancy", split="train",
                 img_width=256, ngram_n=3, max_ngram_features=5000,
                 train_ratio=0.8, val_ratio=0.1, seed=42):
        super().__init__()
        self.data_root = data_root
        self.vuln_type = vuln_type
        self.img_width = img_width

        # 加载预处理缓存（若存在）
        cache_path = os.path.join(data_root, f"{vuln_type}_{split}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            return

        # 否则从原始数据构建
        all_samples = self._load_raw(data_root, vuln_type)
        indices = list(range(len(all_samples)))
        train_idx, test_idx = train_test_split(indices, test_size=1 - train_ratio,
                                               random_state=seed)
        val_idx, test_idx = train_test_split(test_idx,
                                             test_size=0.5, random_state=seed)
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        self.samples = [all_samples[i] for i in split_map[split]]

        os.makedirs(data_root, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.samples, f)

    # ----------------------------------------------------------
    def _load_raw(self, data_root, vuln_type):
        """
        从 data_root 下读取各模态预处理结果。
        目录结构示例：
            data_root/
              cpg/          *.pt  (torch_geometric Data)
              cfg/          *.pt
              bytecode_img/ *.npy
              ngram/        *.npy
              labels.json
        """
        label_file = os.path.join(data_root, "labels.json")
        with open(label_file) as f:
            label_dict = json.load(f)   # {contract_id: {vuln_type: 0/1}}

        samples = []
        for cid, label_info in label_dict.items():
            label = label_info.get(vuln_type, 0)

            cpg_path = os.path.join(data_root, "cpg", f"{cid}.pt")
            cfg_path = os.path.join(data_root, "cfg", f"{cid}.pt")
            img_path = os.path.join(data_root, "bytecode_img", f"{cid}.npy")
            ngram_path = os.path.join(data_root, "ngram", f"{cid}.npy")

            # 任意一个缺失则跳过
            if not all(os.path.exists(p)
                       for p in [cpg_path, cfg_path, img_path, ngram_path]):
                continue

            samples.append({
                "id": cid,
                "cpg_path": cpg_path,
                "cfg_path": cfg_path,
                "img_path": img_path,
                "ngram_path": ngram_path,
                "label": label,
            })
        return samples

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cpg = torch.load(s["cpg_path"])
        cfg = torch.load(s["cfg_path"])
        img = torch.tensor(np.load(s["img_path"]), dtype=torch.float32).unsqueeze(0)
        ngram = torch.tensor(np.load(s["ngram_path"]), dtype=torch.float32)
        label = torch.tensor(s["label"], dtype=torch.long)
        return cpg, cfg, img, ngram, label


def detection_collate_fn(batch):
    """自定义 collate，处理图数据"""
    cpg_list, cfg_list, img_list, ngram_list, label_list = zip(*batch)
    cpg_batch = Batch.from_data_list(list(cpg_list))
    cfg_batch = Batch.from_data_list(list(cfg_list))
    imgs = torch.stack(img_list)
    ngrams = torch.stack(ngram_list)
    labels = torch.stack(label_list)
    return cpg_batch, cfg_batch, imgs, ngrams, labels


def get_detection_loaders(data_root, vuln_type="reentrancy",
                          batch_size=128, num_workers=4, seed=42):
    train_set = SmartContractDetectionDataset(
        data_root, vuln_type, "train", seed=seed)
    val_set   = SmartContractDetectionDataset(
        data_root, vuln_type, "val",   seed=seed)
    test_set  = SmartContractDetectionDataset(
        data_root, vuln_type, "test",  seed=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              collate_fn=detection_collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              collate_fn=detection_collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              collate_fn=detection_collate_fn)
    return train_loader, val_loader, test_loader


# ────────────────────────────────────────────────────────────
# 第四章 定位任务数据集
# ────────────────────────────────────────────────────────────

class ReentrancyLocalizationDataset(Dataset):
    """
    重入漏洞定位数据集 (MANDO + BJUT_SC01)
    每个样本：
      - cfg_graph : CFG图，节点带有 node_label (0/1) 表示是否为漏洞节点
      - label     : 合约级标签 (此任务全为1——重入漏洞合约)
    """

    def __init__(self, data_root, split="train",
                 train_ratio=0.8, seed=42):
        super().__init__()
        cache_path = os.path.join(data_root, f"localization_{split}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            return

        all_samples = self._load_raw(data_root)
        indices = list(range(len(all_samples)))
        train_idx, test_idx = train_test_split(
            indices, test_size=1 - train_ratio, random_state=seed)
        val_idx, test_idx = train_test_split(
            test_idx, test_size=0.5, random_state=seed)
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        self.samples = [all_samples[i] for i in split_map[split]]

        os.makedirs(data_root, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.samples, f)

    def _load_raw(self, data_root):
        cfg_dir = os.path.join(data_root, "cfg_localization")
        samples = []
        for fname in sorted(os.listdir(cfg_dir)):
            if fname.endswith(".pt"):
                samples.append(os.path.join(cfg_dir, fname))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        graph = torch.load(self.samples[idx])
        return graph


def get_localization_loaders(data_root, batch_size=128,
                             num_workers=4, seed=42):
    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_set = ReentrancyLocalizationDataset(data_root, "train", seed=seed)
    val_set   = ReentrancyLocalizationDataset(data_root, "val",   seed=seed)
    test_set  = ReentrancyLocalizationDataset(data_root, "test",  seed=seed)

    train_loader = PyGDataLoader(train_set, batch_size=batch_size,
                                 shuffle=True,  num_workers=num_workers)
    val_loader   = PyGDataLoader(val_set,   batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    test_loader  = PyGDataLoader(test_set,  batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader