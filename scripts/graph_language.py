# -*- coding: utf-8 -*-
"""
图查询语言（GQL）DSL → Cypher 翻译器

"创造图的语言"——将简化中文/英文关键词编译为标准 Cypher 查询语句。
支持交互式命令行和脚本调用两种模式。

用法:
    python graph_language.py                          # 交互式
    python graph_language.py "所有节点"                # 单次翻译
    python graph_language.py --exec "邻居 OF 100"      # 翻译并执行（需Bolt连接）
"""

import sys
import re


# ==================== Token 类型 ====================
KW_ALL = "ALL"
KW_NODES = "NODES"
KW_EDGES = "EDGES"
KW_WHERE = "WHERE"
KW_LABEL = "LABEL"
KW_OF = "OF"
KW_FROM = "FROM"
KW_TO = "TO"
KW_NEIGHBORS = "NEIGHBORS"
KW_PATH = "PATH"
KW_SHORTEST = "SHORTEST"
KW_ALGO = "ALGO"
KW_STATS = "STATS"
KW_HELP = "HELP"
KW_LIMIT = "LIMIT"


# ==================== 命令注册表 ====================
COMMANDS = {
    # 中文关键词 → (命令类型, Cypher模板)
    "所有节点": ("query", "MATCH (n) RETURN n LIMIT {limit}"),
    "所有边": ("query", "MATCH ()-[r]->() RETURN r LIMIT {limit}"),
    "全部节点": ("query", "MATCH (n) RETURN n LIMIT {limit}"),
    "全部边": ("query", "MATCH ()-[r]->() RETURN r LIMIT {limit}"),
    "ALL NODES": ("query", "MATCH (n) RETURN n LIMIT {limit}"),
    "ALL EDGES": ("query", "MATCH ()-[r]->() RETURN r LIMIT {limit}"),
    "统计": ("query", "MATCH (n) RETURN count(n) AS node_count"),
    "STATS": ("query", "MATCH (n) RETURN count(n) AS node_count"),
    "帮助": ("help", None),
    "HELP": ("help", None),
}


def tokenize(text):
    """简单词法分析：将输入文本切分为 token 列表"""
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        # 字符串字面量
        if text[i] == '"' or text[i] == "'":
            quote = text[i]
            j = i + 1
            while j < len(text) and text[j] != quote:
                j += 1
            tokens.append(("STRING", text[i + 1:j]))
            i = j + 1
            continue
        # 数字
        if text[i].isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            tokens.append(("NUMBER", int(text[i:j])))
            i = j
            continue
        # 等号
        if text[i] == "=":
            tokens.append(("EQ", "="))
            i += 1
            continue
        # 标识符/关键词
        if text[i].isalpha() or text[i] == "_":
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("ID", text[i:j]))
            i = j
            continue
        i += 1
    return tokens


def parse(tokens):
    """语法分析：token 序列 → AST 命令节点"""
    if not tokens:
        return {"type": "help", "message": "空输入"}

    text = " ".join(str(t[1]) for t in tokens)

    # 精确匹配内置命令
    upper = text.upper()
    for cmd, (cmd_type, template) in COMMANDS.items():
        if upper == cmd.upper():
            return {"type": cmd_type, "command": cmd, "template": template}

    # WHERE 子句: 节点 WHERE key=value
    where_match = re.match(
        r"(?i)(节点|NODE)\s+WHERE\s+(\w+)\s*=\s*(.+)", text
    )
    if where_match:
        entity = where_match.group(1)
        key = where_match.group(2)
        value = where_match.group(3).strip().strip('"').strip("'")
        label_match = re.match(r"(?i)label\s*=\s*(\w+)", f"{key}={value}")
        if label_match:
            return {
                "type": "query",
                "cypher": f"MATCH (n:{label_match.group(1)}) RETURN n LIMIT 100"
            }
        return {
            "type": "query",
            "cypher": f'MATCH (n {{{key}: "{value}"}}) RETURN n LIMIT 100'
        }

    # 邻居查询: 邻居 OF vid
    neighbor_match = re.match(
        r"(?i)(邻居|NEIGHBORS?)\s+(OF\s+)?(\d+)", text
    )
    if neighbor_match:
        vid = int(neighbor_match.group(3))
        return {
            "type": "query",
            "cypher": f"MATCH (n)-[]-(m) WHERE id(n)={vid} RETURN m LIMIT 100"
        }

    # 路径查询: 路径 FROM vid1 TO vid2
    path_match = re.match(
        r"(?i)(路径|PATH)\s+FROM\s+(\d+)\s+TO\s+(\d+)", text
    )
    if path_match:
        src = int(path_match.group(2))
        dst = int(path_match.group(3))
        return {
            "type": "query",
            "cypher": f"MATCH p=(a)-[*1..6]-(b) WHERE id(a)={src} AND id(b)={dst} RETURN p LIMIT 10"
        }

    # 最短路径: 最短路径 FROM vid1 TO vid2
    shortest_match = re.match(
        r"(?i)(最短路径|SHORTEST\s+PATH)\s+FROM\s+(\d+)\s+TO\s+(\d+)", text
    )
    if shortest_match:
        src = int(shortest_match.group(2))
        dst = int(shortest_match.group(3))
        return {
            "type": "query",
            "cypher": (
                f"MATCH (a), (b) WHERE id(a)={src} AND id(b)={dst} "
                f"WITH a, b CALL algo.shortestPath(a, b) "
                f"YIELD nodeCount, totalCost, path "
                f"RETURN nodeCount, totalCost, path"
            )
        }

    # 算法调用: 算法 algo_name key=value ...
    algo_match = re.match(r"(?i)(算法|ALGO|ALGORITHM)\s+(\w+)\s*(.*)", text)
    if algo_match:
        algo_name = algo_match.group(2).upper()
        params_str = algo_match.group(3)
        params = {}
        if params_str:
            for m in re.finditer(r"(\w+)\s*=\s*([^\s]+)", params_str):
                key, val = m.group(1), m.group(2)
                try:
                    params[key] = int(val)
                except ValueError:
                    try:
                        params[key] = float(val)
                    except ValueError:
                        params[key] = val.strip('"').strip("'")
        return {
            "type": "algorithm",
            "algorithm": algo_name,
            "params": params,
        }

    # 按标签查询: 节点 label=XXX / NODE label=XXX
    label_match = re.match(
        r"(?i)(节点|NODE|NODES?)\s+label\s*=\s*(\w+)", text
    )
    if label_match:
        label = label_match.group(2)
        return {
            "type": "query",
            "cypher": f"MATCH (n:{label}) RETURN n LIMIT 50"
        }

    # 默认：当作 Cypher 直接执行
    if any(kw in upper for kw in ["MATCH", "CREATE", "RETURN", "WITH",
                                    "MERGE", "DELETE", "SET", "REMOVE"]):
        return {"type": "raw_cypher", "cypher": text}

    return {"type": "unknown", "input": text,
            "hint": "无法识别, 试试: 所有节点 / 邻居 OF 100 / 路径 FROM 1 TO 10 / 算法 PAGERANK top_k=10"}


def compile_to_cypher(ast, limit=100):
    """AST → Cypher 编译"""
    if ast["type"] == "query":
        cypher = ast.get("cypher", ast.get("template", "MATCH (n) RETURN n LIMIT 100"))
        return cypher.replace("{limit}", str(limit))

    if ast["type"] == "raw_cypher":
        return ast["cypher"]

    if ast["type"] in ("help", "unknown"):
        return None

    return None


def execute(ast, bolt_client=None, rest_client=None):
    """执行 DSL 命令并返回结果"""
    cmd_type = ast["type"]

    if cmd_type == "query" or cmd_type == "raw_cypher":
        cypher = compile_to_cypher(ast)
        if cypher and bolt_client:
            return bolt_client.execute_query(cypher)
        return {"cypher": cypher, "results": None}

    if cmd_type == "algorithm":
        if rest_client:
            plugin_name = f"{ast['algorithm'].lower()}_plugin"
            return rest_client.call_plugin(plugin_name, ast["params"])
        return {"algorithm": ast["algorithm"], "params": ast["params"],
                "status": "no client"}

    if cmd_type == "help":
        return {"help": get_help_text()}

    if cmd_type == "unknown":
        return {"error": ast.get("hint", "无法识别")}

    return ast


def get_help_text():
    """返回帮助文本"""
    return """
图查询语言（GQL）支持的命令:

=== 图查询 ===
  所有节点 / ALL NODES              → 返回所有节点
  所有边 / ALL EDGES                → 返回所有边
  节点 WHERE name="张三"             → 按属性查找节点
  节点 label=Person                 → 按标签查找节点
  邻居 OF 100 / NEIGHBORS OF 100    → 查找邻居节点
  路径 FROM 100 TO 200              → 查找两点间路径
  最短路径 FROM 100 TO 200          → 计算最短路径
  统计 / STATS                      → 统计节点数

=== 图算法 ===
  算法 PageRank top_k=10 max_iter=20   → 执行 PageRank
  算法 LPA max_iter=20                 → 执行标签传播
  算法 LCC top_k=20                    → 执行局部聚类系数
  算法 Node2Vec p=1.0 q=1.0           → 执行 Node2Vec

=== Cypher 直通 ===
  直接输入 MATCH / CREATE 等标准 Cypher 语句即可执行

=== 其他 ===
  帮助 / HELP    → 显示此帮助
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GQL 图查询语言翻译器")
    parser.add_argument("input", nargs="?", help="DSL 输入文本")
    parser.add_argument("--exec", action="store_true", help="尝试执行")
    args = parser.parse_args()

    if args.input:
        tokens = tokenize(args.input)
        ast = parse(tokens)
        cypher = compile_to_cypher(ast)
        print(f"[DSL]  {args.input}")
        print(f"[AST]  {ast['type']}")
        if cypher:
            print(f"[Cypher] {cypher}")
        elif ast["type"] == "algorithm":
            print(f"[算法] {ast['algorithm']} {ast['params']}")
        elif ast["type"] == "help":
            print(get_help_text())
        else:
            print(f"[结果] {ast}")
    else:
        # 交互式模式
        print("GQL 图查询语言 (输入 '帮助' 获取帮助, '退出' 退出)")
        while True:
            try:
                text = input("\nGQL> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in ("退出", "exit", "quit", "q"):
                break
            tokens = tokenize(text)
            ast = parse(tokens)
            cypher = compile_to_cypher(ast)
            if ast["type"] == "help":
                print(get_help_text())
            elif cypher:
                print(f"[Cypher] {cypher}")
            elif ast["type"] == "algorithm":
                print(f"[算法调用] {ast['algorithm']}")
                print(f"[参数] {json.dumps(ast['params'], indent=2)}")
            else:
                print(f"[{ast['type']}] {ast.get('hint', ast.get('message', ''))}")


if __name__ == "__main__":
    import json
    main()
