# cython: language_level=3, cpp_locals=True, boundscheck=False, wraparound=False
# distutils: language=c++
# -*- coding: utf-8 -*-
"""
标签传播算法（LPA）Cython 插件（改编自老师 ceshi_lpa.py）
支持参数: {"max_iter": 20}
"""

import json
import random
import cython


def Process(db, input):
    """TuGraph Python Plugin 入口（Cython 编译为动态库）"""
    data = json.loads(input)
    max_iter = int(data.get("max_iter", 20))
    random.seed(42)

    txn = db.CreateReadTxn()

    # 收集全图顶点ID
    vids = []
    it = txn.GetVertexIterator()
    while it.IsValid():
        vids.append(it.GetId())
        it.Next()

    N = len(vids)
    if N == 0:
        txn.Abort()
        return (True, json.dumps({"error": "graph is empty", "communities": []}))

    # 初始化标签：每个节点自身ID为标签
    labels = {vid: vid for vid in vids}

    # LPA 核心迭代
    for iteration in range(max_iter):
        changed = False
        new_labels = labels.copy()

        for vid in vids:
            v = txn.GetVertexIterator(vid)
            if not v.IsValid():
                continue

            label_freq = {}
            edge_it = v.GetOutEdgeIterator()
            while edge_it.IsValid():
                dst_vid = edge_it.GetDst()
                if dst_vid in labels:
                    lbl = labels[dst_vid]
                    label_freq[lbl] = label_freq.get(lbl, 0) + 1
                edge_it.Next()

            if label_freq:
                max_count = max(label_freq.values())
                candidates = [
                    lbl for lbl, cnt in label_freq.items()
                    if cnt == max_count
                ]
                new_label = random.choice(candidates)

                if new_label != labels[vid]:
                    new_labels[vid] = new_label
                    changed = True

        labels = new_labels
        if not changed:
            break

    txn.Abort()

    # 整理社区结构
    communities = {}
    for vid, cid in labels.items():
        communities.setdefault(cid, []).append(vid)

    community_list = [
        {"community_id": cid, "size": len(members), "members": members}
        for cid, members in communities.items()
    ]
    community_list.sort(key=lambda x: x["size"], reverse=True)

    return (True, json.dumps({
        "algorithm": "LPA",
        "params": {"max_iter": max_iter},
        "iterations": iteration + 1,
        "total_nodes": N,
        "num_communities": len(community_list),
        "communities": community_list
    }))
