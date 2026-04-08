"""
特征预处理脚本
  - build_cpg()          : 从 Solidity 源码构建 CPG 图 (torch_geometric Data)
  - build_bytecode_img() : 字节码 → 灰度图像 (.npy)
  - build_ngram()        : 操作码 N-gram 频率向量 (.npy)
  - build_cfg()          : 操作码 → CFG 图 (torch_geometric Data)
  - preprocess_all()     : 批量处理入口
"""

import os
import re
import json
import subprocess
import numpy as np
import torch
from collections import Counter, defaultdict
from gensim.models import Word2Vec
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────
# 操作码归一化规则（表3-1）
# ─────────────────────────────────────────────────────────

OPCODE_ABSTRACTION = {}

for i in range(1, 33):
    OPCODE_ABSTRACTION[f"PUSH{i}"] = "PUSH"
for i in range(1, 17):
    OPCODE_ABSTRACTION[f"SWAP{i}"] = "SWAP"
    OPCODE_ABSTRACTION[f"DUP{i}"]  = "DUP"
for i in range(1, 5):
    OPCODE_ABSTRACTION[f"LOG{i}"] = "LOG"
for op in ["LT", "GT", "SLT", "SGT", "EQ", "ISZERO"]:
    OPCODE_ABSTRACTION[op] = "ComparisonOP"
for op in ["AND", "OR", "XOR", "NOT", "BYTE"]:
    OPCODE_ABSTRACTION[op] = "LogicalOP"
for op in ["ADDRESS", "BALANCE", "ORIGIN", "CALLER"]:
    OPCODE_ABSTRACTION[op] = "EnInfoOP"

# 控制转移 / 终止类操作码
BRANCH_OPS  = {"JUMP", "JUMPI"}
TERMINAL_OPS = {"STOP", "RETURN", "REVERT", "SELFDESTRUCT", "INVALID"}


def normalize_opcode(op: str) -> str:
    return OPCODE_ABSTRACTION.get(op, op)


# ─────────────────────────────────────────────────────────
# 1. CPG 构建
# ─────────────────────────────────────────────────────────

def build_cpg(sol_path: str, w2v_model: Word2Vec, out_path: str):
    """
    从 Solidity 源文件构建代码属性图并保存为 torch_geometric Data。

    简化实现：使用 py-solidity-parser 解析 AST，
    手动添加控制语义边和数据流边。
    """
    try:
        from solidity_parser import parser as sol_parser
    except ImportError:
        raise ImportError("请安装: pip install python-solidity-parser")

    # 解析 AST
    ast = sol_parser.parse_file(sol_path, loc=False)
    nodes, edges, edge_types = [], [], []

    def _extract_nodes(node, parent_id=None):
        if not isinstance(node, dict):
            return
        ntype = node.get("type", "Unknown")
        nname = node.get("name", "")
        nid   = len(nodes)
        nodes.append({"type": ntype, "name": nname})
        if parent_id is not None:
            edges.append((parent_id, nid))
            edge_types.append(0)  # 语法边
        for val in node.values():
            if isinstance(val, dict):
                _extract_nodes(val, nid)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        _extract_nodes(item, nid)

    _extract_nodes(ast)

    # 控制语义边：FunctionDefinition → FunctionCall
    func_def_ids   = [i for i, n in enumerate(nodes) if n["type"] == "FunctionDefinition"]
    func_call_ids  = [i for i, n in enumerate(nodes) if n["type"] == "FunctionCall"]
    for fd in func_def_ids:
        for fc in func_call_ids:
            edges.append((fd, fc))
            edge_types.append(1)  # 控制语义边

    # 数据流边：VariableDeclaration → Assignment
    var_decl_ids   = [i for i, n in enumerate(nodes) if n["type"] == "VariableDeclaration"]
    assignment_ids = [i for i, n in enumerate(nodes) if n["type"] == "ExpressionStatement"]
    for vd in var_decl_ids:
        for asn in assignment_ids:
            edges.append((vd, asn))
            edge_types.append(2)  # 数据流边

    if len(nodes) == 0:
        return

    # 节点特征：Word2Vec 编码
    feat_dim = w2v_model.vector_size
    x = np.zeros((len(nodes), feat_dim), dtype=np.float32)
    for i, n in enumerate(nodes):
        token = n["type"] + "_" + n["name"] if n["name"] else n["type"]
        if token in w2v_model.wv:
            x[i] = w2v_model.wv[token]

    x_t       = torch.tensor(x, dtype=torch.float32)
    edge_idx  = torch.tensor(edges, dtype=torch.long).t().contiguous() \
                 if edges else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x_t, edge_index=edge_idx, edge_type=edge_type)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(data, out_path)


# ─────────────────────────────────────────────────────────
# 2. 字节码灰度图像
# ─────────────────────────────────────────────────────────

def build_bytecode_img(bytecode_hex: str, out_path: str, width: int = 256):
    """
    bytecode_hex: 十六进制字节码字符串（不带0x前缀）
    """
    bytecode_hex = bytecode_hex.replace("0x", "").strip()
    if len(bytecode_hex) % 2 != 0:
        bytecode_hex += "0"
    byte_vals = np.array([int(bytecode_hex[i:i+2], 16)
                          for i in range(0, len(bytecode_hex), 2)],
                         dtype=np.uint8)
    # 补全至 width 的整数倍
    rem = len(byte_vals) % width
    if rem != 0:
        byte_vals = np.concatenate([byte_vals,
                                    np.zeros(width - rem, dtype=np.uint8)])
    img = byte_vals.reshape(-1, width).astype(np.float32) / 255.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, img)


# ─────────────────────────────────────────────────────────
# 3. 操作码 N-gram 频率向量
# ─────────────────────────────────────────────────────────

def disassemble_bytecode(bytecode_hex: str) -> list:
    """简单反汇编：返回操作码列表（归一化后）"""
    bytecode_hex = bytecode_hex.replace("0x", "").strip()
    opcodes = []
    i = 0
    byte_list = bytes.fromhex(bytecode_hex)
    while i < len(byte_list):
        op = byte_list[i]
        # PUSH1~PUSH32: 0x60~0x7f
        if 0x60 <= op <= 0x7f:
            push_bytes = op - 0x5f
            opcodes.append(normalize_opcode(f"PUSH{push_bytes}"))
            i += 1 + push_bytes
        else:
            # 查表（简化：使用编号）
            name = _OPCODE_TABLE.get(op, f"UNKNOWN_{op:02x}")
            opcodes.append(normalize_opcode(name))
            i += 1
    return opcodes


# 简化的 EVM 操作码表（部分）
_OPCODE_TABLE = {
    0x00: "STOP",    0x01: "ADD",    0x02: "MUL",    0x03: "SUB",
    0x04: "DIV",     0x05: "SDIV",   0x06: "MOD",    0x07: "SMOD",
    0x08: "ADDMOD",  0x09: "MULMOD", 0x0a: "EXP",
    0x10: "LT",      0x11: "GT",     0x12: "SLT",    0x13: "SGT",
    0x14: "EQ",      0x15: "ISZERO", 0x16: "AND",    0x17: "OR",
    0x18: "XOR",     0x19: "NOT",
    0x20: "SHA3",
    0x30: "ADDRESS", 0x31: "BALANCE",0x32: "ORIGIN", 0x33: "CALLER",
    0x34: "CALLVALUE",0x35: "CALLDATALOAD",0x36: "CALLDATASIZE",
    0x50: "POP",     0x51: "MLOAD",  0x52: "MSTORE", 0x54: "SLOAD",
    0x55: "SSTORE",  0x56: "JUMP",   0x57: "JUMPI",  0x58: "PC",
    0x5b: "JUMPDEST",
    0xf0: "CREATE",  0xf1: "CALL",   0xf2: "CALLCODE",
    0xf3: "RETURN",  0xf4: "DELEGATECALL",
    0xfd: "REVERT",  0xff: "SELFDESTRUCT",
}
for i in range(0x80, 0x90):
    _OPCODE_TABLE[i] = f"DUP{i - 0x7f}"
for i in range(0x90, 0xa0):
    _OPCODE_TABLE[i] = f"SWAP{i - 0x8f}"
for i in range(0xa0, 0xa5):
    _OPCODE_TABLE[i] = f"LOG{i - 0xa0}"


def build_ngram_vector(opcodes: list, vocab: dict, n: int = 3) -> np.ndarray:
    """
    opcodes: 归一化后的操作码列表
    vocab  : {ngram_tuple: index}（全数据集共享词表）
    """
    counter = Counter(
        tuple(opcodes[j:j+n]) for j in range(len(opcodes) - n + 1)
    )
    vec = np.zeros(len(vocab), dtype=np.float32)
    for gram, cnt in counter.items():
        if gram in vocab:
            vec[vocab[gram]] += cnt
    # TF 归一化
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def build_ngram_vocab(all_opcodes_list: list, n: int = 3,
                       max_features: int = 5000) -> dict:
    """构建全局 N-gram 词表"""
    counter = Counter()
    for opcodes in all_opcodes_list:
        for j in range(len(opcodes) - n + 1):
            counter[tuple(opcodes[j:j+n])] += 1
    most_common = counter.most_common(max_features)
    return {gram: idx for idx, (gram, _) in enumerate(most_common)}


# ─────────────────────────────────────────────────────────
# 4. 操作码 CFG 构建
# ─────────────────────────────────────────────────────────

def build_cfg_graph(opcodes: list, w2v_model: Word2Vec, out_path: str,
                    node_labels: list = None):
    """
    opcodes    : 归一化操作码列表
    node_labels: 可选，每个基本块节点的漏洞标签 (0/1)，用于定位任务
    """
    # ---- 基本块划分 ----
    leaders = {0}
    for i, op in enumerate(opcodes):
        if op in BRANCH_OPS | TERMINAL_OPS:
            if i + 1 < len(opcodes):
                leaders.add(i + 1)
        # JUMPDEST 是可能的跳转目标
        if op == "JUMPDEST":
            leaders.add(i)

    leaders_sorted = sorted(leaders)
    blocks = []
    for k, start in enumerate(leaders_sorted):
        end = leaders_sorted[k + 1] if k + 1 < len(leaders_sorted) else len(opcodes)
        # 寻找块的实际结束位置
        actual_end = start
        for i in range(start, end):
            actual_end = i + 1
            if opcodes[i] in BRANCH_OPS | TERMINAL_OPS:
                break
        blocks.append(opcodes[start:actual_end])

    n_blocks = len(blocks)
    if n_blocks == 0:
        return

    # ---- 控制流边连接 ----
    edges = []
    for k, block in enumerate(blocks):
        if not block:
            continue
        last_op = block[-1]
        if last_op == "JUMP":
            # 无条件跳转：跳转目标未知，保守连接到所有 JUMPDEST 块
            for j, b in enumerate(blocks):
                if b and b[0] == "JUMPDEST":
                    edges.append((k, j))
        elif last_op == "JUMPI":
            # 条件跳转：跳转目标 + 顺序后继
            if k + 1 < n_blocks:
                edges.append((k, k + 1))
            for j, b in enumerate(blocks):
                if b and b[0] == "JUMPDEST":
                    edges.append((k, j))
        elif last_op in TERMINAL_OPS:
            pass  # 无后继
        else:
            if k + 1 < n_blocks:
                edges.append((k, k + 1))

    # ---- 节点特征（Word2Vec 均值池化） ----
    feat_dim = w2v_model.vector_size
    x = np.zeros((n_blocks, feat_dim), dtype=np.float32)
    for i, block in enumerate(blocks):
        vecs = [w2v_model.wv[op]
                for op in block if op in w2v_model.wv]
        if vecs:
            x[i] = np.mean(vecs, axis=0)

    x_t = torch.tensor(x, dtype=torch.float32)
    if edges:
        edge_idx = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_idx = torch.zeros((2, 0), dtype=torch.long)

    kwargs = {"x": x_t, "edge_index": edge_idx}
    if node_labels is not None:
        # 对齐 node_labels 到基本块
        nl = np.zeros(n_blocks, dtype=np.int64)
        for i, start in enumerate(sorted(leaders)):
            if i < len(node_labels):
                nl[i] = node_labels[i]
        kwargs["node_label"] = torch.tensor(nl, dtype=torch.long)

    data = Data(**kwargs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(data, out_path)


# ─────────────────────────────────────────────────────────
# 5. 批量预处理入口
# ─────────────────────────────────────────────────────────

def preprocess_all(raw_dir: str, out_dir: str,
                   ngram_n: int = 3, max_ngram_features: int = 5000,
                   w2v_size: int = 128, w2v_window: int = 5):
    """
    raw_dir 目录结构：
        contracts/  *.sol          Solidity 源文件
        bytecode/   *.hex          字节码十六进制文件
        labels.json                {id: {vuln_type: 0/1}}
    """
    sol_dir   = os.path.join(raw_dir, "contracts")
    byte_dir  = os.path.join(raw_dir, "bytecode")
    label_file= os.path.join(raw_dir, "labels.json")

    with open(label_file) as f:
        labels = json.load(f)

    contract_ids = list(labels.keys())

    # ---- 反汇编所有字节码 ----
    print("[1/5] 反汇编字节码...")
    all_opcodes = {}
    for cid in contract_ids:
        hex_path = os.path.join(byte_dir, f"{cid}.hex")
        if not os.path.exists(hex_path):
            continue
        with open(hex_path) as f:
            hex_str = f.read().strip()
        all_opcodes[cid] = disassemble_bytecode(hex_str)

    # ---- 训练 Word2Vec ----
    print("[2/5] 训练 Word2Vec...")
    corpus = [ops for ops in all_opcodes.values() if ops]
    # 也加入 AST 节点 token（简化：仅用操作码语料）
    w2v = Word2Vec(sentences=corpus, vector_size=w2v_size,
                   window=w2v_window, min_count=1, workers=4, epochs=10)
    w2v.save(os.path.join(out_dir, "word2vec.model"))

    # ---- 构建 N-gram 词表 ----
    print("[3/5] 构建 N-gram 词表...")
    vocab = build_ngram_vocab(list(all_opcodes.values()),
                              n=ngram_n, max_features=max_ngram_features)
    with open(os.path.join(out_dir, "ngram_vocab.json"), "w") as f:
        json.dump({str(k): v for k, v in vocab.items()}, f)

    # ---- 生成各模态特征 ----
    print("[4/5] 提取多模态特征...")
    for cid in contract_ids:
        # 字节码灰度图
        hex_path = os.path.join(byte_dir, f"{cid}.hex")
        if os.path.exists(hex_path):
            with open(hex_path) as f:
                hex_str = f.read().strip()
            build_bytecode_img(
                hex_str,
                os.path.join(out_dir, "bytecode_img", f"{cid}.npy"))

            # N-gram 向量
            if cid in all_opcodes:
                vec = build_ngram_vector(all_opcodes[cid], vocab, n=ngram_n)
                np.save(os.path.join(out_dir, "ngram", f"{cid}.npy"), vec)

            # CFG 图
            if cid in all_opcodes:
                build_cfg_graph(
                    all_opcodes[cid], w2v,
                    os.path.join(out_dir, "cfg", f"{cid}.pt"))

        # CPG 图
        sol_path = os.path.join(sol_dir, f"{cid}.sol")
        if os.path.exists(sol_path):
            try:
                build_cpg(sol_path, w2v,
                          os.path.join(out_dir, "cpg", f"{cid}.pt"))
            except Exception as e:
                print(f"  CPG 构建失败 [{cid}]: {e}")

    # ---- 拷贝 labels ----
    import shutil
    shutil.copy(label_file, os.path.join(out_dir, "labels.json"))
    print("[5/5] 预处理完成！输出目录:", out_dir)


# ─────────────────────────────────────────────────────────
# 独立运行
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",  default="data/raw")
    p.add_argument("--out_dir",  default="data/detection")
    p.add_argument("--ngram_n",  type=int, default=3)
    p.add_argument("--max_ngram",type=int, default=5000)
    p.add_argument("--w2v_size", type=int, default=128)
    args = p.parse_args()
    preprocess_all(args.raw_dir, args.out_dir,
                   ngram_n=args.ngram_n,
                   max_ngram_features=args.max_ngram,
                   w2v_size=args.w2v_size)