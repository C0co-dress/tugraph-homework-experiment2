# -*- coding: utf-8 -*-
import json
import random

def Process(db, input):
    # 1. 解析输入参数（支持 max_iter 控制最大迭代次数）
    raw_data = input
    parsed_data = json.loads(raw_data)
    max_iter = int(parsed_data.get("max_iter", 20))
    random.seed(42)  # 固定随机种子，保证每次运行结果可复现

    # 2. 创建只读事务
    txn = db.CreateReadTxn()

    # 收集全图顶点ID
    vids = []
    it = txn.GetVertexIterator()
    while it.IsValid():
        vids.append(it.GetId())
        it.Next()

    # 初始化标签：每个节点初始独立为一个社区（标签=自身ID）
    labels = {vid: vid for vid in vids}

    # 3. 标签传播(LPA)核心逻辑
    for _ in range(max_iter):
        changed = False
        new_labels = labels.copy()  # 同步更新：本迭代使用上一轮标签状态

        for vid in vids:
            # 获取当前顶点对象（注：TuGraph 4.x 精确获取顶点应用 GetVertex(vid)）
            v = txn.GetVertexIterator(vid)
            if not v.IsValid():
                continue

            label_freq = {}
            # 遍历当前顶点的出边迭代器，统计邻居社区标签频次
            edge_it = v.GetOutEdgeIterator()
            while edge_it.IsValid():
                dst_vid = edge_it.GetDst()
                if dst_vid in labels:
                    lbl = labels[dst_vid]
                    label_freq[lbl] = label_freq.get(lbl, 0) + 1
                edge_it.Next()

            if label_freq:
                # 找出邻居中出现次数最多的标签
                max_count = max(label_freq.values())
                # 收集所有达到最高频的标签（用于随机打破平局）
                candidates = [lbl for lbl, cnt in label_freq.items() if cnt == max_count]
                new_label = random.choice(candidates)

                if new_label != labels[vid]:
                    new_labels[vid] = new_label
                    changed = True

        labels = new_labels
        if not changed:
            break  # 标签不再变化，算法提前收敛

    # 4. 释放只读事务资源（只读查询推荐用 Abort）
    #txn.Abort()

    # 5. 返回结果（TuGraph 插件标准格式：成功标志, 结果字符串）
    # 结果格式: [[顶点ID, 所属社区ID], ...]
    result = [[txn.GetVertexIterator(vid).GetField("name"), cid] for vid, cid in labels.items()]
    txn.Abort()
    return (True, str(result))
