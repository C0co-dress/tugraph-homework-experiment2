# -*- coding: utf-8 -*-
"""
YAGO 图数据导入脚本
基于 Neo4j Bolt 驱动向 TuGraph 批量导入节点和边数据。
YAGO 数据模型: person, movie, genre, keyword, user 节点类型
                produce, acted_in, direct, write, has_genre, has_keyword, rate 边类型
参考: 老师PPT第8页 YAGO 示例数据模型
"""

import os
import csv
from neo4j import GraphDatabase

# ==================== 连接配置 ====================
URI = os.environ.get("TUGRAPH_URI", "bolt://localhost:7687")
AUTH = (os.environ.get("TUGRAPH_USER", "admin"),
        os.environ.get("TUGRAPH_PASSWORD", "73@TuGraph"))
DATABASE = os.environ.get("TUGRAPH_DATABASE", "default")


def create_schema(session):
    """创建图模型: 5种节点类型 + 8种边类型"""
    # 节点类型通过 Cypher CREATE 语句隐式定义
    # TuGraph 支持动态创建标签
    print("[1/4] 图模型: person / movie / genre / keyword / user (5种节点)")
    print("        produce / acted_in / direct / write / has_genre / has_keyword / rate (8种边)")
    # TuGraph 在导入时自动识别标签，无需显式 CREATE LABEL
    # 验证连接
    result = session.run("MATCH (n) RETURN count(n) AS cnt")
    existing = result.single()["cnt"]
    print(f"        当前数据库已有 {existing} 个节点")


def import_persons(session, csv_path):
    """导入 person 节点"""
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session.run(
                "CREATE (p:person {id: $id, name: $name, born: $born})",
                id=int(row.get("id", 0)),
                name=row.get("name", ""),
                born=int(row.get("born", 0)) if row.get("born") else None,
            )
            count += 1
    print(f"        导入 person: {count} 条")
    return count


def import_movies(session, csv_path):
    """导入 movie 节点"""
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session.run(
                "CREATE (m:movie {id: $id, title: $title, tagline: $tagline, "
                "duration: $duration, rated: $rated})",
                id=int(row.get("id", 0)),
                title=row.get("title", ""),
                tagline=row.get("tagline", ""),
                duration=int(row.get("duration", 0)) if row.get("duration") else None,
                rated=row.get("rated", ""),
            )
            count += 1
    print(f"        导入 movie: {count} 条")
    return count


def import_genres(session, csv_path):
    """导入 genre 节点"""
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session.run(
                "CREATE (g:genre {id: $id, name: $name})",
                id=int(row.get("id", 0)),
                name=row.get("name", ""),
            )
            count += 1
    print(f"        导入 genre: {count} 条")
    return count


def import_keywords(session, csv_path):
    """导入 keyword 节点"""
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session.run(
                "CREATE (k:keyword {id: $id, name: $name})",
                id=int(row.get("id", 0)),
                name=row.get("name", ""),
            )
            count += 1
    print(f"        导入 keyword: {count} 条")
    return count


def import_users(session, csv_path):
    """导入 user 节点"""
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session.run(
                "CREATE (u:user {id: $id, name: $name})",
                id=int(row.get("id", 0)),
                name=row.get("name", ""),
            )
            count += 1
    print(f"        导入 user: {count} 条")
    return count


def import_edges(session, csv_path, edge_label, src_label, dst_label,
                 src_col="src_id", dst_col="dst_id"):
    """通用边导入"""
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cypher = (
                f"MATCH (a:{src_label} {{id: $src_id}}), "
                f"(b:{dst_label} {{id: $dst_id}}) "
                f"CREATE (a)-[:{edge_label}]->(b)"
            )
            session.run(
                cypher,
                src_id=int(row.get(src_col, 0)),
                dst_id=int(row.get(dst_col, 0)),
            )
            count += 1
    print(f"        导入 {edge_label}: {count} 条")
    return count


def import_rated_edges(session, csv_path):
    """导入 rate 边(带 score 属性)"""
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session.run(
                "MATCH (u:user {id: $user_id}), (m:movie {id: $movie_id}) "
                "CREATE (u)-[:rate {score: $score}]->(m)",
                user_id=int(row.get("user_id", 0)),
                movie_id=int(row.get("movie_id", 0)),
                score=float(row.get("score", 0)),
            )
            count += 1
    print(f"        导入 rate: {count} 条")
    return count


def verify(session):
    """验证导入结果"""
    print("\n[4/4] 验证导入结果:")
    r = session.run("MATCH (n) RETURN count(n) AS total").single()
    print(f"        节点总数: {r['total']}")
    r = session.run("MATCH ()-[r]->() RETURN count(r) AS total").single()
    print(f"        边总数: {r['total']}")
    # 按标签统计
    for label in ["person", "movie", "genre", "keyword", "user"]:
        r = session.run(
            f"MATCH (n:{label}) RETURN count(n) AS cnt"
        ).single()
        print(f"        {label}: {r['cnt']}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")

    print(f"连接 TuGraph: {URI}")
    print(f"数据库: {DATABASE}")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("连接成功!\n")

        with driver.session(database=DATABASE) as session:
            create_schema(session)

            print("\n[2/4] 导入节点数据...")
            import_persons(session,
                           os.path.join(data_dir, "person.csv"))
            import_movies(session,
                          os.path.join(data_dir, "movie.csv"))
            import_genres(session,
                          os.path.join(data_dir, "genre.csv"))
            import_keywords(session,
                            os.path.join(data_dir, "keyword.csv"))
            import_users(session,
                         os.path.join(data_dir, "user.csv"))

            print("\n[3/4] 导入边数据...")
            import_edges(session,
                         os.path.join(data_dir, "produce.csv"),
                         "produce", "person", "movie",
                         "person_id", "movie_id")
            import_edges(session,
                         os.path.join(data_dir, "acted_in.csv"),
                         "acted_in", "person", "movie",
                         "person_id", "movie_id")
            import_edges(session,
                         os.path.join(data_dir, "direct.csv"),
                         "direct", "person", "movie",
                         "person_id", "movie_id")
            import_edges(session,
                         os.path.join(data_dir, "write.csv"),
                         "write", "person", "movie",
                         "person_id", "movie_id")
            import_edges(session,
                         os.path.join(data_dir, "has_genre.csv"),
                         "has_genre", "movie", "genre",
                         "movie_id", "genre_id")
            import_edges(session,
                         os.path.join(data_dir, "has_keyword.csv"),
                         "has_keyword", "movie", "keyword",
                         "movie_id", "keyword_id")
            import_rated_edges(session,
                               os.path.join(data_dir, "rate.csv"))

            verify(session)

    print("\n数据导入完成!")


if __name__ == "__main__":
    main()
