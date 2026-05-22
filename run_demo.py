# -*- coding: utf-8 -*-
"""
一键演示脚本: 数据导入 → 随机游走采样 → Node2Vec 训练 → 输出结果
用于实验二的第2、3步——Node2Vec 数据导入与完整算法演示

用法:
    python run_demo.py              # 完整流程
    python run_demo.py --sample-only  # 仅随机游走采样
    python run_demo.py --train-only   # 仅训练（需已有采样数据）
"""

import json
import os
import sys
import random
import math
import argparse

# 添加 scripts 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from neo4j import GraphDatabase


# ==================== 配置 ====================
URI = os.environ.get("TUGRAPH_URI", "bolt://localhost:7687")
AUTH = (os.environ.get("TUGRAPH_USER", "admin"),
        os.environ.get("TUGRAPH_PASSWORD", "73@TuGraph"))
DATABASE = os.environ.get("TUGRAPH_DATABASE", "Movie")

# Node2Vec 参数
WALK_LEN = 10
NUM_WALKS = 5
MAX_STARTS = 20
P = 1.0
Q = 1.0
VEC_DIM = 64
EPOCHS = 3
LR = 0.05
NEG_SAMPLES = 5
WINDOW = 3
RANDOM_SEED = 42

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def query(session, cypher, **params):
    """执行 Cypher 并返回 list[dict]"""
    return session.run(cypher, **params).data()


def load_graph(session, max_starts):
    """从 TuGraph 读取图结构"""
    print("[1/5] 读取图结构...")

    node_count = query(session, "MATCH (n) RETURN count(n) AS cnt")[0]["cnt"]
    edge_count = query(session, "MATCH ()-[r]->() RETURN count(r) AS cnt")[0]["cnt"]
    print(f"      节点数: {node_count}, 边数: {edge_count}")

    # 获取有出边的节点
    start_rows = query(
        session,
        f"MATCH (n)-[r]->(m) RETURN id(n) AS vid, count(r) AS deg "
        f"ORDER BY deg DESC, vid ASC LIMIT {max_starts}"
    )
    start_nodes = [int(r["vid"]) for r in start_rows]

    # 构建邻接表
    edge_rows = query(
        session,
        "MATCH (n)-[r]->(m) RETURN id(n) AS src, id(m) AS dst"
    )
    adjacency = {}
    for row in edge_rows:
        src = int(row["src"])
        dst = int(row["dst"])
        adjacency.setdefault(src, []).append(dst)
        adjacency.setdefault(dst, adjacency.get(dst, []))

    print(f"      选取起点: {len(start_nodes)} 个, 邻接表: {len(adjacency)} 个节点")
    return adjacency, start_nodes, node_count, edge_count


def node2vec_walk(adjacency, start_node, walk_length, p, q):
    """一条 Node2Vec 二阶偏置随机游走"""
    walk = [start_node]
    previous = None
    current = start_node

    for _ in range(walk_length - 1):
        neighbors = adjacency.get(current, [])
        if not neighbors:
            break

        if previous is None:
            next_node = random.choice(neighbors)
        else:
            prev_nbrs = set(adjacency.get(previous, []))
            weights = []
            for n in neighbors:
                if n == previous:
                    weights.append(1.0 / p)
                elif n in prev_nbrs:
                    weights.append(1.0)
                else:
                    weights.append(1.0 / q)
            next_node = random.choices(neighbors, weights=weights, k=1)[0]

        walk.append(next_node)
        previous = current
        current = next_node
    return walk


def generate_walks(adjacency, start_nodes, walk_len, num_walks, p, q):
    """从多个起点生成游走"""
    print("\n[2/5] 生成 Node2Vec 随机游走...")
    walks = []
    for s in start_nodes:
        for _ in range(num_walks):
            walks.append(node2vec_walk(adjacency, s, walk_len, p, q))
    print(f"      生成 {len(walks)} 条游走路径")
    for i, w in enumerate(walks[:3]):
        print(f"      示例 {i+1}: {w}")
    return walks


def train_embeddings(walks, vec_dim, epochs, lr, neg_samples, window):
    """纯 Python Skip-gram 负采样训练"""
    print("\n[3/5] 训练 Node2Vec 嵌入向量...")

    vocab = sorted(list(set(n for w in walks for n in w)))
    vid2idx = {v: i for i, v in enumerate(vocab)}
    V = len(vocab)
    print(f"      词表大小: {V}, 嵌入维度: {vec_dim}, 训练轮数: {epochs}")

    if V == 0:
        return {}

    random.seed(RANDOM_SEED)
    W_in = [[random.gauss(0, 0.1) for _ in range(vec_dim)] for _ in range(V)]
    W_out = [[0.0] * vec_dim for _ in range(V)]
    neg_table = [random.randint(0, V - 1) for _ in range(V * max(neg_samples, 1))]

    def sigmoid(x):
        if x > 20: return 1.0
        if x < -20: return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    for epoch in range(epochs):
        cur_lr = lr * (1.0 - epoch / max(epochs, 1))
        loss_sum = 0.0
        for walk in walks:
            for i in range(len(walk)):
                ci = vid2idx[walk[i]]
                w = random.randint(1, window)
                for j in range(max(0, i - w), min(len(walk), i + w + 1)):
                    if i == j: continue
                    ti = vid2idx[walk[j]]

                    # 正样本
                    dot = sum(a * b for a, b in zip(W_in[ci], W_out[ti]))
                    sig = sigmoid(dot)
                    grad = (1.0 - sig) * cur_lr
                    old_in = W_in[ci][:]
                    for d in range(vec_dim):
                        W_in[ci][d] += grad * W_out[ti][d]
                        W_out[ti][d] += grad * old_in[d]

                    # 负样本
                    for _ in range(neg_samples):
                        ni = neg_table[random.randint(0, len(neg_table) - 1)]
                        if ni == ti: continue
                        dot = sum(a * b for a, b in zip(W_in[ci], W_out[ni]))
                        sig = sigmoid(dot)
                        grad = (0.0 - sig) * cur_lr
                        old_in2 = W_in[ci][:]
                        for d in range(vec_dim):
                            W_in[ci][d] += grad * W_out[ni][d]
                            W_out[ni][d] += grad * old_in2[d]

        print(f"      epoch {epoch + 1}/{epochs} 完成")

    embeddings = {str(v): [round(x, 4) for x in W_in[i]]
                  for v, i in vid2idx.items()}
    return embeddings


def save_outputs(walks, embeddings, node_count, edge_count):
    """保存结果到文件"""
    print("\n[4/5] 保存输出...")

    walks_file = os.path.join(OUTPUT_DIR, "node2vec_walks.json")
    emb_file = os.path.join(OUTPUT_DIR, "node2vec_embeddings.json")
    summary_file = os.path.join(OUTPUT_DIR, "node2vec_summary.txt")

    with open(walks_file, "w", encoding="utf-8") as f:
        json.dump(walks, f, ensure_ascii=False, indent=2)
    print(f"      游走结果: {walks_file}")

    with open(emb_file, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    print(f"      嵌入向量: {emb_file}")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Node2Vec 训练摘要\n")
        f.write(f"节点数: {node_count}\n")
        f.write(f"边数: {edge_count}\n")
        f.write(f"游走条数: {len(walks)}\n")
        f.write(f"嵌入节点数: {len(embeddings)}\n")
        f.write(f"嵌入维度: {VEC_DIM}\n")
        f.write(f"\n参数: p={P}, q={Q}, walk_len={WALK_LEN}, "
                f"num_walks={NUM_WALKS}, max_starts={MAX_STARTS}\n")
        f.write(f"训练: epochs={EPOCHS}, lr={LR}, "
                f"neg_samples={NEG_SAMPLES}, window={WINDOW}\n")
    print(f"      摘要: {summary_file}")


def show_results(embeddings):
    """展示结果"""
    print("\n[5/5] 嵌入向量结果:")
    items = list(embeddings.items())
    for vid, vec in items[:3]:
        print(f"      vid={vid}")
        print(f"      向量(前8维): {vec[:8]}")
        print(f"      向量范数: {round(math.sqrt(sum(x*x for x in vec)), 4)}")
    avg_norm = sum(
        math.sqrt(sum(x * x for x in v)) for v in embeddings.values()
    ) / max(len(embeddings), 1)
    print(f"\n      总嵌入节点数: {len(embeddings)}")
    print(f"      平均向量范数: {round(avg_norm, 4)}")


def main():
    parser = argparse.ArgumentParser(description="Node2Vec 一键演示")
    parser.add_argument("--sample-only", action="store_true",
                        help="仅随机游走采样")
    parser.add_argument("--train-only", action="store_true",
                        help="仅训练（从已有游走文件读取）")
    args = parser.parse_args()

    print("=" * 60)
    print("  Node2Vec 图嵌入算法演示")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print(f"连接 TuGraph 成功: {URI}\n")

        with driver.session(database=DATABASE) as session:
            if args.train_only:
                walks_file = os.path.join(OUTPUT_DIR, "node2vec_walks.json")
                if not os.path.exists(walks_file):
                    print(f"错误: 找不到游走文件 {walks_file}")
                    sys.exit(1)
                with open(walks_file, "r") as f:
                    walks = json.load(f)
                node_count = query(session,
                                   "MATCH (n) RETURN count(n) AS cnt")[0]["cnt"]
                edge_count = query(session,
                                   "MATCH ()-[r]->() RETURN count(r) AS cnt")[0]["cnt"]
                print(f"从文件加载 {len(walks)} 条游走路径")
            else:
                adjacency, start_nodes, node_count, edge_count = load_graph(
                    session, MAX_STARTS)

                if not start_nodes:
                    print("错误: 图中没有有出边的节点, 请先导入数据!")
                    sys.exit(1)

                walks = generate_walks(adjacency, start_nodes,
                                       WALK_LEN, NUM_WALKS, P, Q)

                if args.sample_only:
                    save_outputs(walks, {}, node_count, edge_count)
                    print("\n随机游走采样完成!")
                    return

            embeddings = train_embeddings(walks, VEC_DIM, EPOCHS, LR,
                                          NEG_SAMPLES, WINDOW)
            save_outputs(walks, embeddings, node_count, edge_count)
            show_results(embeddings)

    print("\n" + "=" * 60)
    print("  演示完成! 截图以下内容用于实验报告:")
    print(f"  1. 终端输出 (本窗口)")
    print(f"  2. {os.path.join(OUTPUT_DIR, 'node2vec_walks.json')}")
    print(f"  3. {os.path.join(OUTPUT_DIR, 'node2vec_embeddings.json')}")
    print(f"  4. {os.path.join(OUTPUT_DIR, 'node2vec_summary.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
