# cython: language_level=3, boundscheck=False
# -*- coding: utf-8 -*-
"""
BFS 广度优先遍历 TuGraph 存储过程
从指定起点出发，按层次从左到右依次访问所有可达节点。
入口: Process(db, input)
输入: {"start_vid": 1}
输出: {"start_vid": 1, "visited_count": N, "bfs_order": [vid1, vid2, ...]}
"""

import json


def Process(db, input):
    data = json.loads(input)
    src = data["start_vid"]

    txn = db.CreateReadTxn()
    visited = []
    queue = [src]
    visited_set = {src}

    while queue:
        vid = queue.pop(0)
        visited.append(vid)
        it = txn.GetVertexIterator(vid)
        if it.IsValid():
            eit = it.GetOutEdgeIterator()
            while eit.IsValid():
                nxt = eit.GetDst()
                if nxt not in visited_set:
                    visited_set.add(nxt)
                    queue.append(nxt)
                eit.Next()

    txn.Abort()
    return (True, json.dumps({
        "start_vid": src,
        "visited_count": len(visited),
        "bfs_order": visited
    }))
