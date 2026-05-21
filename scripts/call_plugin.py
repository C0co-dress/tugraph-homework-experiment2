# -*- coding: utf-8 -*-
"""
TuGraph REST API 通用插件调用脚本
基于老师 ceshi_requests_post.py / ceshi_requests_get.py 模式

用途: 从本地机器调用远程 TuGraph 的 REST API
      - 登录获取 JWT token
      - 上传 Python 插件（存储过程）
      - 调用插件并获取执行结果
      - 列出/删除已有插件

用法:
    python call_plugin.py --list                          # 列出所有插件
    python call_plugin.py --load pagerank_plugin.py       # 上传插件
    python call_plugin.py --call pagerank_plugin '{"max_iter":20}'  # 调用
"""

import json
import os
import sys
import argparse
import requests


# ==================== 配置 ====================
BASE_URL = os.environ.get("TUGRAPH_BASE_URL", "http://localhost:7070")
AUTH = {
    "user": os.environ.get("TUGRAPH_USER", "admin"),
    "password": os.environ.get("TUGRAPH_PASSWORD", "73@TuGraph"),
}
HEADERS = {
    "Accept": "application/json; charset=UTF-8",
    "Content-Type": "application/json; charset=UTF-8",
}


def login():
    """登录获取 JWT token"""
    url = f"{BASE_URL}/login"
    resp = requests.post(url, json=AUTH, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("jwt", "")
    if token:
        HEADERS["Authorization"] = f"Bearer {token}"
        print(f"[OK] 登录成功, JWT: {token[:20]}...")
    else:
        print(f"[WARN] 未获取到 JWT, 响应: {data}")
    return token


def list_plugins():
    """列出已有存储过程"""
    url = f"{BASE_URL}/db/default/python_plugin"
    resp = requests.get(url, headers=HEADERS, auth=(AUTH["user"], AUTH["password"]))
    resp.raise_for_status()
    data = resp.json()
    print("[插件列表]")
    if isinstance(data, list):
        for p in data:
            print(f"  - {p}")
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def load_plugin(plugin_path, plugin_name=None):
    """上传 Python 存储过程到 TuGraph"""
    if plugin_name is None:
        plugin_name = os.path.splitext(os.path.basename(plugin_path))[0]

    with open(plugin_path, "r", encoding="utf-8") as f:
        code = f.read()

    url = f"{BASE_URL}/db/default/python_plugin"
    payload = {
        "name": plugin_name,
        "code": code,
        "type": "PY",
    }
    resp = requests.post(url, json=payload, headers=HEADERS,
                         auth=(AUTH["user"], AUTH["password"]))
    resp.raise_for_status()
    result = resp.json()
    print(f"[OK] 插件 '{plugin_name}' 上传成功")
    print(f"     响应: {json.dumps(result, ensure_ascii=False)[:200]}")
    return result


def call_plugin(plugin_name, params=None):
    """调用存储过程"""
    url = f"{BASE_URL}/db/default/python_plugin/{plugin_name}"
    if params is None:
        params = {}
    payload = {"data": json.dumps(params)}

    resp = requests.post(url, json=payload, headers=HEADERS,
                         auth=(AUTH["user"], AUTH["password"]))
    resp.raise_for_status()
    result = resp.json()
    print(f"[结果] {plugin_name}:")
    # 解包结果
    if isinstance(result, list) and len(result) == 2:
        success, data_str = result[0], result[1]
        try:
            data = json.loads(data_str)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, TypeError):
            print(data_str)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def delete_plugin(plugin_name):
    """删除存储过程"""
    url = f"{BASE_URL}/db/default/python_plugin/{plugin_name}"
    resp = requests.delete(url, headers=HEADERS,
                           auth=(AUTH["user"], AUTH["password"]))
    resp.raise_for_status()
    print(f"[OK] 插件 '{plugin_name}' 已删除")


def main():
    parser = argparse.ArgumentParser(description="TuGraph REST API 插件管理")
    parser.add_argument("--list", action="store_true", help="列出所有插件")
    parser.add_argument("--load", type=str, metavar="PATH", help="上传插件")
    parser.add_argument("--name", type=str, help="插件名称（默认取文件名）")
    parser.add_argument("--call", type=str, metavar="NAME", help="调用插件")
    parser.add_argument("--params", type=str, default="{}", help="JSON 参数")
    parser.add_argument("--delete", type=str, metavar="NAME", help="删除插件")
    parser.add_argument("--login-only", action="store_true", help="仅测试登录")

    args = parser.parse_args()

    # 登录
    login()

    if args.login_only:
        return

    if args.list:
        list_plugins()
    elif args.load:
        load_plugin(args.load, args.name)
    elif args.call:
        params = json.loads(args.params)
        call_plugin(args.call, params)
    elif args.delete:
        delete_plugin(args.delete)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python call_plugin.py --list")
        print("  python call_plugin.py --load ../plugins/pagerank_plugin.py")
        print("  python call_plugin.py --call pagerank_plugin "
              "--params '{\"max_iter\":20,\"top_k\":10}'")


if __name__ == "__main__":
    main()
