# 智能合约漏洞检测及定位实验代码

本仓库对应论文《智能合约漏洞检测及定位算法研究》第三章和第四章的实验实现。

---

## 目录结构

```
smart_contract/
├── preprocess/
│   └── feature_extractor.py   # 多模态特征预处理（CPG/字节码图/N-gram/CFG）
├── models/
│   ├── detection_model.py     # 第三章：多模态检测模型
│   └── localization_model.py  # 第四章：对抗+嵌入对齐定位模型
├── data/
│   └── dataset.py             # Dataset / DataLoader 封装
├── train.py                   # 训练脚本
├── evaluate.py                # 评估、消融、可视化脚本
├── run_experiments.py         # 主实验入口
└── requirements.txt
```

---

## 环境依赖

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=1.8.0
torch-geometric>=2.0.0
scikit-learn>=0.24
gensim>=4.0
numpy
matplotlib
python-solidity-parser
```

---

## 快速开始

### 1. 数据预处理

```bash
python preprocess/feature_extractor.py \
    --raw_dir  data/raw \
    --out_dir  data/detection \
    --ngram_n  3 \
    --max_ngram 5000 \
    --w2v_size  128
```

`data/raw` 目录结构：
```
data/raw/
  contracts/  *.sol       Solidity 源文件
  bytecode/   *.hex       字节码十六进制
  labels.json             {"contract_id": {"reentrancy": 1, ...}}
```

### 2. 第三章：多模态检测模型训练

```bash
python run_experiments.py \
    --task detection \
    --data_root  data/detection \
    --vuln_type  reentrancy \
    --epochs     50 \
    --batch_size 128 \
    --lr         0.001 \
    --hidden_dim 256
```

关键超参数（对应论文表3-3）：

| 参数 | 值 |
|------|-----|
| 模态特征向量维度 | 256 |
| RGCN 层数 | 3 |
| DGCN 层数 | 3 |
| 学习率 | 0.001 |
| batch size | 128 |
| epoch | 50 |
| dropout 率 | 0.2 |

### 3. 第四章：漏洞定位模型训练

```bash
python run_experiments.py \
    --task         localization \
    --loc_data_root data/localization \
    --beta         0.3 \
    --lam          0.2 \
    --n_clusters   6 \
    --epochs       50
```

关键超参数（对应论文表4-2）：

| 参数 | 值 |
|------|-----|
| 节点特征向量维度 | 256 |
| λ（嵌入对齐权重）| 0.2 |
| β（对抗约束权重）| 0.3 |
| 学习率 | 0.001 |
| 锚点数量 K | 6 |
| dropout 率 | 0.3 |

### 4. 完整流水线

```bash
python run_experiments.py --task all
```

### 5. 仅评估

```bash
python run_experiments.py --task eval
```

---

## 实验说明

### 第三章实验

| 实验 | 脚本函数 | 输出文件 |
|------|----------|----------|
| DGCN×RGCN 层数热力图 | `exp_layer_heatmap()` | `results/heatmap1.png` |
| 学习率敏感性 | `exp_lr_detection()` | `results/f1_lr.png` |
| 消融实验 | `exp_ablation_detection()` | 终端输出 |
| 注意力权重分析 | `analyze_attention_weights()` | 终端输出 |

### 第四章实验

| 实验 | 脚本函数 | 输出文件 |
|------|----------|----------|
| (λ,β) 权重热力图 | `exp_weight_heatmap()` | `results/heatmap.png` |
| 锚点数量分析 | `exp_anchor_num()` | `results/anchor_para.png` |
| 保真度分析 | `fidelity_analysis()` | `results/fidelity.png` |
| 消融实验 | `exp_ablation_localization()` | `results/ablation1.png` |
| 时间分析 | `time_analysis()` | `results/time.png` |

---

## 模型架构

### 第三章：MultiModalDetector

```
源码 CPG → RGCNConv × 3 → global_mean_pool
字节码   → Conv + ResBlock × 2 + SPP     ┐
N-gram   → MLP (5000→256)               ├→ 两级注意力融合 → FC → 分类
CFG      → DGCNConv × 3 → global_mean_pool ┘
```

### 第四章：VulnLocalizor

```
CFG → [冻结 CFGEncoder] → 节点嵌入
                         ↓
                   MaskGenerator (MLP)
                   ↙          ↘
              掩码子图        补集子图
               ↓                ↓
         [冻结 Classifier]  [冻结 Classifier]
               ↓                ↓
           L_pred            L_uni (KL)
                              +
              子图嵌入 → 锚点距离统计 → L_align
                              ↓
              L = α·L_pred + β·L_uni + λ·L_align
```

---

## 数据集

- **检测任务**：SmartBugs-Wild + SolidFI-Benchmark（共 4676 个合约）
  https://github.com/smartbugs/smartbugs-wild
  https://github.com/DependableSystemsLab/SolidiFI-benchmark
- **定位任务**：MANDO + BJUT_SC01（2800 个重入漏洞合约）