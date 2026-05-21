# cython: language_level=3, cpp_locals=True, boundscheck=False, wraparound=False
# distutils: language=c++
# -*- coding: utf-8 -*-
"""
PageRank Cython 插件（改编自老师 ceshi_pagerank.py）
支持参数: {"max_iter": 20, "damping": 0.85, "top_k": 10}
"""

import json
import cython


def Process(db, input):
    """TuGraph Python Plugin 入口（Cython 编译为动态库）"""
    data = json.loads(input)
    max_iter = int(data.get("max_iter", 20))
    damping = float(data.get("damping", 0.85))
    top_k = int(data.get("top_k", 10))

    txn = db.CreateReadTxn()

    # 收集顶点ID和出度
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
        return (True, json.dumps({"error": "graph is empty", "results": []}))

    # 初始化 PageRank 值（均匀分布）
    pr = {vid: 1.0 / N for vid in vids}

    # 迭代计算 PageRank（Push 模式，与老师逻辑一致）
    for iteration in range(max_iter):
        new_pr = {vid: (1.0 - damping) / N for vid in vids}

        # 计算悬挂节点总权重
        dangling_mass = sum(
            pr[vid] for vid in vids if out_deg[vid] == 0
        ) * damping

        # 遍历顶点，推送权重
        for vid in vids:
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

        # 悬挂节点权重均分
        if dangling_mass > 0:
            equal_share = dangling_mass / N
            for vid in vids:
                new_pr[vid] += equal_share

        # 收敛检查
        diff = sum(abs(new_pr[vid] - pr[vid]) for vid in vids)
        pr = new_pr
        if diff < 1e-6:
            break

    txn.Abort()

    # 排序返回 Top-K
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    top_result = sorted_pr[:top_k]
    result_list = [[vid, round(score, 6)] for vid, score in top_result]

    return (True, json.dumps({
        "algorithm": "PageRank",
        "params": {"max_iter": max_iter, "damping": damping, "top_k": top_k},
        "iterations": iteration + 1,
        "total_nodes": N,
        "results": result_list
    }))
