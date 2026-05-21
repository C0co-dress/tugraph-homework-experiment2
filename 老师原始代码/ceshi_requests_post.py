# -*- coding: utf-8 -*-

import requests

url = "http://localhost:7070/login"
headers = {
    "Accept": "application/json; charset=UTF-8",
    "Content-Type": "application/json; charset=UTF-8"
}
payload = {
    "user": "admin",
    "password": "73@TuGraph"
}

# 发送 POST 请求
response = requests.post(url, json=payload, headers=headers)

# 打印响应状态码与返回内容
print(f"状态码: {response.status_code}")
print(f"响应数据: {response.json()}")



