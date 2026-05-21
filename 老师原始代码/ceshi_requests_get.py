# -*- coding: utf-8 -*-

import requests

url = "http://localhost:7070/db/follow/python_plugin"
headers = {
    "Accept": "application/json; charset=UTF-8"
}

# 发送 GET 请求
response = requests.get(url, headers=headers, auth=("admin", "73@TuGraph"))

# 打印状态码与解析后的 JSON 数据
print(f"状态码: {response.status_code}")
print(f"响应数据: {response.json()}")




