# cython: language_level=3, cpp_locals=True, boundscheck=False, wraparound=False
# distutils: language=c++
# -*- coding: utf-8 -*-
"""
Node2Vec 算法 Cython 插件（改编自老师 ceshi_node2vec_5.py）
流程: 全图采样起点 → 偏置随机游走 → Skip-gram 负采样训练 → 输出嵌入向量
支持参数: {"p": 1.0, "q": 1.0, "walk_len": 10, "num_walks": 5,
          "max_starts": 50, "vector_size": 64, "epochs": 3,
          "learning_rate": 0.05, "neg_samples": 5, "window": 3}
"""

import json
import random
import math
import cython


def Process(db, input):
    """TuGraph Python Plugin 入口"""
    data = json.loads(input)

    # === 参数解析（带默认值） ===
    p = float(data.get("p", 1.0))
    q = float(data.get("q", 1.0))
    walk_len = int(data.get("walk_len", 10))
    num_walks = int(data.get("num_walks", 5))
    max_starts = int(data.get("max_starts", 50))
    vec_dim = int(data.get("vector_size", 64))
    epochs = int(data.get("epochs", 3))
    lr = float(data.get("learning_rate", 0.05))
    neg_samples = int(data.get("neg_samples", 5))
    window = int(data.get("window", 3))
    random.seed(42)

    # === 阶段1: 读取图结构 ===
    txn = db.CreateReadTxn()
    neighbor_cache = {}

    def get_neighbors(vid):
        """带缓存的邻居获取（避免重复数据库访问）"""
        if vid not in neighbor_cache:
            v = txn.GetVertexIterator(vid)
            nbrs = []
            if v.IsValid():
                eit = v.GetOutEdgeIterator()
                while eit.IsValid():
                    nbrs.append(eit.GetDst())
                    eit.Next()
            neighbor_cache[vid] = nbrs
        return neighbor_cache[vid]

    # 获取全图顶点并随机采样起点
    all_vids = []
    it = txn.GetVertexIterator()
    while it.IsValid():
        all_vids.append(it.GetId())
        it.Next()

    if not all_vids:
        txn.Abort()
        return (True, json.dumps({"error": "graph is empty", "embeddings": {}}))

    N = len(all_vids)
    start_nodes = random.sample(all_vids, min(N, max_starts))

    # === 阶段2: Node2Vec 偏置随机游走 ===
    def biased_walk(start_node, length):
        """一条 Node2Vec 二阶偏置随机游走"""
        walk = [start_node]
        curr = start_node
        prev = start_node

        for _ in range(length - 1):
            nbrs = get_neighbors(curr)
            if not nbrs:
                break

            prev_nbrs_set = set(get_neighbors(prev))
            weights = []
            for n in nbrs:
                if n == prev:
                    weights.append(1.0 / p)           # Return
                elif n in prev_nbrs_set:
                    weights.append(1.0)               # In-Out (BFS)
                else:
                    weights.append(1.0 / q)           # Explore (DFS)

            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            next_node = random.choices(nbrs, weights=probs, k=1)[0]

            walk.append(next_node)
            prev = curr
            curr = next_node
        return walk

    # 生成所有游走序列
    all_walks = []
    for s in start_nodes:
        for _ in range(num_walks):
            all_walks.append(biased_walk(s, walk_len))

    txn.Abort()  # 游走完成后释放事务

    # === 阶段3: Skip-gram 负采样训练（纯 Python） ===
    vocab = sorted(list(set(n for w in all_walks for n in w)))
    vid2idx = {vid: i for i, vid in enumerate(vocab)}
    vocab_size = len(vocab)

    if vocab_size == 0:
        return (True, json.dumps({"error": "no walks generated", "embeddings": {}}))

    # 初始化嵌入矩阵
    W_in = [[random.gauss(0, 0.1) for _ in range(vec_dim)]
            for _ in range(vocab_size)]
    W_out = [[0.0] * vec_dim for _ in range(vocab_size)]
    neg_table = [random.randint(0, vocab_size - 1)
                 for _ in range(vocab_size * max(neg_samples, 1))]

    def sigmoid(x):
        """数值稳定的 sigmoid"""
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    # 训练循环
    total_walks = len(all_walks)
    for epoch in range(epochs):
        cur_lr = lr * (1.0 - epoch / max(epochs, 1))
        for walk in all_walks:
            for i in range(len(walk)):
                center_idx = vid2idx[walk[i]]
                w = random.randint(1, window)
                for j in range(max(0, i - w), min(len(walk), i + w + 1)):
                    if i == j:
                        continue
                    target_idx = vid2idx[walk[j]]

                    # 正样本更新 (label=1)
                    dot = sum(a * b for a, b in
                              zip(W_in[center_idx], W_out[target_idx]))
                    sig = sigmoid(dot)
                    grad = (1.0 - sig) * cur_lr
                    old_in = W_in[center_idx][:]
                    for d in range(vec_dim):
                        W_in[center_idx][d] += grad * W_out[target_idx][d]
                        W_out[target_idx][d] += grad * old_in[d]

                    # 负样本更新 (label=0)
                    for _ in range(neg_samples):
                        neg_idx = neg_table[
                            random.randint(0, len(neg_table) - 1)]
                        if neg_idx == target_idx:
                            continue
                        dot = sum(a * b for a, b in
                                  zip(W_in[center_idx], W_out[neg_idx]))
                        sig = sigmoid(dot)
                        grad = (0.0 - sig) * cur_lr
                        old_in2 = W_in[center_idx][:]
                        for d in range(vec_dim):
                            W_in[center_idx][d] += grad * W_out[neg_idx][d]
                            W_out[neg_idx][d] += grad * old_in2[d]

    # === 阶段4: 提取嵌入向量 ===
    embeddings = {}
    for vid, idx in vid2idx.items():
        embeddings[str(vid)] = [round(v, 4) for v in W_in[idx]]

    return (True, json.dumps({
        "algorithm": "Node2Vec",
        "params": {
            "p": p, "q": q, "walk_len": walk_len, "num_walks": num_walks,
            "max_starts": max_starts, "vector_size": vec_dim, "epochs": epochs,
            "learning_rate": lr, "neg_samples": neg_samples, "window": window
        },
        "total_nodes": N,
        "start_nodes_sampled": len(start_nodes),
        "total_walks": total_walks,
        "vocab_size": vocab_size,
        "embedding_dim": vec_dim,
        "embeddings": embeddings
    }))
