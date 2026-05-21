# -*- coding: utf-8 -*-

# coding=gbk
import json
import random
import math

def Process(db, input):
    # 1. 解析输入参数（带默认值，支持按需调优）
    data = json.loads(input)
    p = float(data.get("p", 1.0))
    q = float(data.get("q", 1.0))
    walk_len = int(data.get("walk_len", 10))
    walks_per_start = int(data.get("num_walks", 5))
    max_starts = int(data.get("max_starts", 50))      # 方案2核心：从全图随机采样的起点数量
    vec_dim = int(data.get("vector_size", 64))
    epochs = int(data.get("epochs", 3))
    lr = float(data.get("learning_rate", 0.05))
    neg_samples = int(data.get("neg_samples", 5))
    window = int(data.get("window", 3))
    random.seed(42)

    # 2. 创建只读事务 & 邻居缓存
    txn = db.CreateReadTxn()
    neighbor_cache = {}

    def get_neighbors(vid):
        if vid not in neighbor_cache:
            # TuGraph 4.x 标准：精准获取顶点使用 GetVertex(vid)
            v = txn.GetVertexIterator(vid)
            nbrs = []
            if v.IsValid():
                eit = v.GetOutEdgeIterator()
                while eit.IsValid():
                    nbrs.append(eit.GetDst())
                    eit.Next()
            neighbor_cache[vid] = nbrs
        return neighbor_cache[vid]

    # 3. 方案2：自动获取全图顶点 ID 并随机采样起点
    all_vids = []
    it = txn.GetVertexIterator()
    while it.IsValid():
        all_vids.append(it.GetId())
        it.Next()
    
    if not all_vids:
        txn.Abort()
        return (True, "{}")

    # 随机采样起点（图较小时全量采样，较大时限制为 max_starts）
    start_nodes = random.sample(all_vids, min(len(all_vids), max_starts))

    # 4. Node2Vec 偏置随机游走核心逻辑
    def biased_walk(start_node, length):
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
                    weights.append(1.0 / p)      # Return
                elif n in prev_nbrs_set:
                    weights.append(1.0)          # In-Out (BFS倾向)
                else:
                    weights.append(1.0 / q)      # Explore (DFS倾向)
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            next_node = random.choices(nbrs, weights=probs, k=1)[0]
            walk.append(next_node)
            prev = curr
            curr = next_node
        return walk

    # 生成游走序列
    all_walks = []
    for s in start_nodes:
        for _ in range(walks_per_start):
            all_walks.append(biased_walk(s, walk_len))

    # 释放数据库事务（训练在内存中进行，无需锁）
    txn.Abort()

    # 5. 构建词表 & 初始化嵌入矩阵（纯 Python 实现）
    vocab = sorted(list(set(n for w in all_walks for n in w)))
    vid2idx = {vid: i for i, vid in enumerate(vocab)}
    vocab_size = len(vocab)
    if vocab_size == 0:
        return (True, "{}")

    #random.seed(42)
    W_in = [[random.gauss(0, 0.1) for _ in range(vec_dim)] for _ in range(vocab_size)]
    W_out = [[0.0] * vec_dim for _ in range(vocab_size)]
    neg_table = [random.randint(0, vocab_size - 1) for _ in range(vocab_size * max(neg_samples, 1))]

    def sigmoid(x):
        if x > 20: return 1.0
        if x < -20: return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    # 6. Skip-gram with Negative Sampling (SGNS) 训练
    for epoch in range(epochs):
        cur_lr = lr * (1.0 - epoch / epochs)  # 线性学习率衰减
        for walk in all_walks:
            for i in range(len(walk)):
                center_idx = vid2idx[walk[i]]
                w = random.randint(1, window)
                # 动态上下文窗口
                for j in range(max(0, i - w), min(len(walk), i + w + 1)):
                    if i == j: continue
                    target_idx = vid2idx[walk[j]]

                    # 🔹 正样本更新
                    dot = sum(a*b for a, b in zip(W_in[center_idx], W_out[target_idx]))
                    sig = sigmoid(dot)
                    grad = (1.0 - sig) * cur_lr
                    old_in = W_in[center_idx][:]
                    for d in range(vec_dim):
                        W_in[center_idx][d] += grad * W_out[target_idx][d]
                        W_out[target_idx][d] += grad * old_in[d]

                    # 🔹 负样本更新
                    for _ in range(neg_samples):
                        neg_idx = neg_table[random.randint(0, len(neg_table)-1)]
                        if neg_idx == target_idx: continue
                        dot = sum(a*b for a, b in zip(W_in[center_idx], W_out[neg_idx]))
                        sig = sigmoid(dot)
                        grad = (0.0 - sig) * cur_lr
                        old_in = W_in[center_idx][:]
                        for d in range(vec_dim):
                            W_in[center_idx][d] += grad * W_out[neg_idx][d]
                            W_out[neg_idx][d] += grad * old_in[d]

    # 7. 提取最终向量并格式化输出
    embeddings = {}
    for vid, idx in vid2idx.items():
        embeddings[str(vid)] = [round(v, 4) for v in W_in[idx]]

    return (True, json.dumps(embeddings))



