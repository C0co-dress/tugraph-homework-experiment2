# coding=gbk
import json
import random

#参数: 起始节点ID=100, 游走长度=10, 游走次数=5, p=1.0, q=1.0
#{"start_vid": 100, "walk_len": 10, "num_walks": 5, "p": 1.0, "q": 1.0}

def Process(db, input):
    # 1. JSON解析输入参数（禁止使用 try，依赖调用方传入合法JSON）
    data = json.loads(input)
    start_vid = int(data["start_vid"])
    walk_len = int(data["walk_len"])
    num_walks = int(data["num_walks"])
    p = float(data["p"])
    q = float(data["q"])

    # 2. 创建只读事务
    txn = db.CreateReadTxn()

    # 邻居缓存字典（避免同一节点在多次游走中重复查库）
    neighbor_cache = {}

    def get_neighbors(vid):
        if vid not in neighbor_cache:
            v = txn.GetVertexIterator(vid)
            nbrs = []
            if v.IsValid():
                edge_it = v.GetOutEdgeIterator()
                while edge_it.IsValid():
                    # 获取目标顶点ID（TuGraph 4.x 标准API）
                    nbrs.append(edge_it.GetDst())
                    edge_it.Next()
            neighbor_cache[vid] = nbrs
        return neighbor_cache[vid]

    # 3. Node2Vec 偏置随机游走核心逻辑
    def biased_walk(start_node, length):
        walk = [start_node]
        curr = start_node
        prev = start_node  # 初始步时 prev 与 curr 相同

        for _ in range(length - 1):
            nbrs = get_neighbors(curr)
            if not nbrs:
                break

            # 获取前驱节点的邻居集合（用于 O(1) 判断是否为 BFS/DFS 倾向）
            prev_nbrs_set = set(get_neighbors(prev))

            # 计算未归一化转移权重（Node2Vec 核心公式）
            weights = []
            for n in nbrs:
                if n == prev:
                    weights.append(1.0 / p)      # 返回上一步 (Return)
                elif n in prev_nbrs_set:
                    weights.append(1.0)          # 局部探索 (BFS倾向)
                else:
                    weights.append(1.0 / q)      # 广度探索 (DFS倾向)

            # 权重归一化
            total_w = sum(weights)
            norm_weights = [w / total_w for w in weights]

            # 按权重随机采样下一个节点
            next_node = random.choices(nbrs, weights=norm_weights, k=1)[0]

            walk.append(next_node)
            prev = curr
            curr = next_node
        return walk

    # 执行指定次数的游走
    all_walks = []
    for _ in range(num_walks):
        all_walks.append(biased_walk(start_vid, walk_len))

    # 4. 释放只读事务资源
    txn.Abort()

    # 5. 返回结果（TuGraph 插件标准格式：成功标志, 结果字符串）
    return (True, str(all_walks))


