# -*- coding: utf-8 -*-

import json
import random
import math

#{"start_vid": 100, "walk_len": 20, "num_walks": 30, "p": 1.0, "q": 1.0, "vector_size": 32, "epochs": 5, "learning_rate": 0.025, "neg_samples": 3}

def Process(db, input):
    # 1. 解析输入参数（支持自定义训练超参）
    data = json.loads(input)
    start_vid = int(data["start_vid"])
    walk_len = int(data["walk_len"])
    num_walks = int(data["num_walks"])
    p = float(data["p"])
    q = float(data["q"])
    
    # Node2Vec 训练参数（带合理默认值）
    vec_dim = int(data.get("vector_size", 64))
    epochs = int(data.get("epochs", 3))
    lr = float(data.get("learning_rate", 0.05))
    neg_samples = int(data.get("neg_samples", 5))
    window = int(data.get("window", 3))

    # 2. 创建只读事务 & 邻居缓存
    txn = db.CreateReadTxn()
    neighbor_cache = {}

    def get_neighbors(vid):
        if vid not in neighbor_cache:
            # ⚠️ TuGraph 4.x 标准 API：精准获取顶点用 GetVertex(vid)
            v = txn.GetVertexIterator(vid)
            nbrs = []
            if v.IsValid():
                edge_it = v.GetOutEdgeIterator()
                while edge_it.IsValid():
                    nbrs.append(edge_it.GetDst())
                    edge_it.Next()
            neighbor_cache[vid] = nbrs
        return neighbor_cache[vid]

    # 3. 偏置随机游走 (Node2Vec 采样核心)
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
                    weights.append(1.0 / p)
                elif n in prev_nbrs_set:
                    weights.append(1.0)
                else:
                    weights.append(1.0 / q)
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            curr = random.choices(nbrs, weights=probs, k=1)[0]
            walk.append(curr)
            prev = walk[-2]
        return walk

    all_walks = [biased_walk(start_vid, walk_len) for _ in range(num_walks)]
    txn.Abort()

    # 4. 构建词表 & 初始化嵌入矩阵 (纯 Python 实现)
    vocab = list(set(n for walk in all_walks for n in walk))
    vid2idx = {vid: i for i, vid in enumerate(vocab)}
    vocab_size = len(vocab)
    if vocab_size == 0:
        return (True, "{}")

    random.seed(42)
    # 输入向量矩阵 W_in，输出向量矩阵 W_out
    W_in = [[random.gauss(0, 0.1) for _ in range(vec_dim)] for _ in range(vocab_size)]
    W_out = [[0.0] * vec_dim for _ in range(vocab_size)]
    
    # 简化版负采样表（均匀分布）
    neg_table = [random.randint(0, vocab_size - 1) for _ in range(vocab_size * neg_samples)]

    def sigmoid(x):
        if x > 20: return 1.0
        if x < -20: return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    # 5. Skip-gram with Negative Sampling (SGNS) 训练
    for epoch in range(epochs):
        current_lr = lr * (1.0 - epoch / epochs)  # 学习率衰减
        for walk in all_walks:
            for i in range(len(walk)):
                center_idx = vid2idx[walk[i]]
                # 动态上下文窗口
                w = random.randint(1, window)
                for j in range(max(0, i - w), min(len(walk), i + w + 1)):
                    if i == j: continue
                    target_idx = vid2idx[walk[j]]
                    
                    # 正样本更新 (label=1)
                    dot = sum(a*b for a, b in zip(W_in[center_idx], W_out[target_idx]))
                    sig = sigmoid(dot)
                    grad = (1 - sig) * current_lr
                    old_in = W_in[center_idx][:]
                    for d in range(vec_dim):
                        W_in[center_idx][d] += grad * W_out[target_idx][d]
                        W_out[target_idx][d] += grad * old_in[d]
                        
                    # 负样本更新 (label=0)
                    for _ in range(neg_samples):
                        neg_idx = neg_table[random.randint(0, len(neg_table)-1)]
                        if neg_idx == target_idx: continue
                        dot = sum(a*b for a, b in zip(W_in[center_idx], W_out[neg_idx]))
                        sig = sigmoid(dot)
                        grad = (0 - sig) * current_lr
                        old_in = W_in[center_idx][:]
                        for d in range(vec_dim):
                            W_in[center_idx][d] += grad * W_out[neg_idx][d]
                            W_out[neg_idx][d] += grad * old_in[d]

    # 6. 提取最终向量（通常取输入矩阵 W_in）并格式化输出
    embeddings = {}
    for vid, idx in vid2idx.items():
        embeddings[str(vid)] = [round(v, 4) for v in W_in[idx]]

    # 7. 返回结果（TuGraph 插件标准格式）
    return (True, json.dumps(embeddings))



