# -*- coding: utf-8 -*-
import json
#{"max_iter": 20, "damping": 0.85, "top_k": 15}

def Process(db, input):
    # 1. 解析输入参数（支持 max_iter, damping, top_k）
    data = json.loads(input)
    max_iter = int(data.get("max_iter", 20))
    damping = float(data.get("damping", 0.85))
    top_k = int(data.get("top_k", 10))

    # 2. 创建只读事务
    txn = db.CreateReadTxn()

    # 3. 收集全图顶点ID并统计出度
    vids = []
    out_deg = {}
    it = txn.GetVertexIterator()
    while it.IsValid():
        vid = it.GetId()
        vids.append(vid)
        deg = 0
        v = txn.GetVertexIterator(vid)
        if v.IsValid():
            eit = v.GetOutEdgeIterator()
            while eit.IsValid():
                deg += 1
                eit.Next()
        out_deg[vid] = deg
        it.Next()

    N = len(vids)
    if N == 0:
        txn.Abort()
        return (True, "[]")

    # 初始化 PageRank 值 (均匀分布)
    pr = {vid: 1.0 / N for vid in vids}

    # 4. 迭代计算 PageRank (Push 模式)
    for _ in range(max_iter):
        # 基础阻尼分配 + 随机跳转概率
        new_pr = {vid: (1.0 - damping) / N for vid in vids}
        
        # 计算悬挂节点（出度为0）的总权重
        dangling_mass = sum(pr[vid] for vid in vids if out_deg[vid] == 0) * damping

        # 遍历所有顶点，将权重沿出边推送给邻居
        it = txn.GetVertexIterator()
        while it.IsValid():
            vid = it.GetId()
            if out_deg[vid] > 0:
                contrib = pr[vid] * damping / out_deg[vid]
                v = txn.GetVertexIterator(vid)
                if v.IsValid():
                    eit = v.GetOutEdgeIterator()
                    while eit.IsValid():
                        dst = eit.GetDst()
                        if dst in new_pr:
                            new_pr[dst] += contrib
                        eit.Next()
            it.Next()

        # 将悬挂节点的权重均匀分配给所有顶点
        if dangling_mass > 0:
            equal_share = dangling_mass / N
            for vid in vids:
                new_pr[vid] += equal_share

        # 检查收敛阈值
        diff = sum(abs(new_pr[vid] - pr[vid]) for vid in vids)
        pr = new_pr
        if diff < 1e-6:
            break

    # 5. 释放只读事务资源
    txn.Abort()

    # 6. 排序并返回 Top-K 结果
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    top_result = sorted_pr[:top_k]
    # 格式化输出: [[vid, score], [vid, score], ...]
    result_list = [[vid, round(score, 6)] for vid, score in top_result]
    return (True, str(result_list))

    return (True, str(top_k))
