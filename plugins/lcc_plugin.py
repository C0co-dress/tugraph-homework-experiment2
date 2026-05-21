# cython: language_level=3, cpp_locals=True, boundscheck=False, wraparound=False
# distutils: language=c++
# -*- coding: utf-8 -*-
"""
局部聚类系数（LCC）Cython 插件
计算每个顶点的局部聚类系数: 邻居之间的实际边数 / 邻居对总数
支持参数: {"top_k": 20} 返回 Top-K 或全部节点的 LCC 值

公式: LCC(v) = 2 * T(v) / (d(v) * (d(v) - 1))
  其中 T(v) = 通过 v 的三角形数, d(v) = v 的度数
"""

import json
import cython


def Process(db, input):
    """TuGraph Python Plugin 入口"""
    data = json.loads(input)
    top_k = int(data.get("top_k", 20))

    txn = db.CreateReadTxn()

    # 收集顶点ID和邻居列表
    vids = []
    neighbors = {}
    it = txn.GetVertexIterator()
    while it.IsValid():
        vid = it.GetId()
        vids.append(vid)
        nbrs = set()
        v = txn.GetVertexIterator(vid)
        if v.IsValid():
            eit = v.GetOutEdgeIterator()
            while eit.IsValid():
                nbrs.add(eit.GetDst())
                eit.Next()
        neighbors[vid] = nbrs
        it.Next()

    N = len(vids)
    if N == 0:
        txn.Abort()
        return (True, json.dumps({"error": "graph is empty", "results": []}))

    txn.Abort()

    # 计算每个顶点的 LCC
    lcc_results = []
    for vid in vids:
        nbrs = neighbors[vid]
        d = len(nbrs)
        if d < 2:
            lcc = 0.0
        else:
            # 统计邻居之间的边数（三角形数）
            triangles = 0
            nbr_list = list(nbrs)
            for i in range(len(nbr_list)):
                for j in range(i + 1, len(nbr_list)):
                    if nbr_list[j] in neighbors.get(nbr_list[i], set()):
                        triangles += 1

            max_possible = d * (d - 1) / 2.0
            lcc = 2.0 * triangles / (d * (d - 1.0))

        lcc_results.append((vid, round(lcc, 6), d, triangles))

    # 度数>=2 的平均 LCC
    valid = [x for x in lcc_results if x[2] >= 2]
    avg_lcc = round(sum(x[1] for x in valid) / len(valid), 6) if valid else 0.0

    # 排序返回 Top-K
    lcc_results.sort(key=lambda x: x[1], reverse=True)
    top = lcc_results[:top_k]

    return (True, json.dumps({
        "algorithm": "LCC",
        "params": {"top_k": top_k},
        "total_nodes": N,
        "average_lcc": avg_lcc,
        "results": [
            {"vid": vid, "lcc": lcc, "degree": deg, "triangles": tri}
            for vid, lcc, deg, tri in top
        ]
    }))
