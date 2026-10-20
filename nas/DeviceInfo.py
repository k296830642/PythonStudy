#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import ipaddress
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

# 全局 Session 以复用连接
session = requests.Session()

FIELDS = [
    "device_id", "device_name", "status", "product_model",
    "active_time", "warranty_expire_time", "ssd_expire_time",
    "network_status", "version"
]

def parse_network(network_input: str) -> List[str]:
    """解析网段，支持数字简写、CIDR、范围、单个IP"""
    if network_input.isdigit():
        network_input = f"192.168.{network_input}.0/24"

    ips = []
    network_input = network_input.strip()

    if '/' in network_input:
        net = ipaddress.ip_network(network_input, strict=False)
        return [str(ip) for ip in net.hosts()]

    if '-' in network_input:
        parts = network_input.split('-')
        start_str, end_str = parts[0].strip(), parts[1].strip()
        if end_str.isdigit():
            end_str = start_str.rsplit('.', 1)[0] + '.' + end_str
        start_int = int(ipaddress.ip_address(start_str))
        end_int = int(ipaddress.ip_address(end_str))
        return [str(ipaddress.ip_address(i)) for i in range(min(start_int, end_int), max(start_int, end_int) + 1)]

    return [network_input]

def scan_ip(ip: str, timeout: float) -> Optional[Dict[str, Any]]:
    """扫描单个IP，带简单重试"""
    url = f"http://{ip}:5502/device/info"
    for _ in range(2):  # 重试一次
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('code') == 200:
                    result = {field: data.get('data', {}).get(field) for field in FIELDS}
                    result['ip'] = ip
                    return result
            break # 成功或明确失败则跳出
        except Exception:
            continue
    return None

def main():
    # 1. 控制台输入
    if len(sys.argv) > 1:
        network_input = sys.argv[1]
    else:
        network_input = input("请输入网段（如 8 表示 192.168.8.0/24，或输入 192.168.1.1-100）: ")

    try:
        ip_list = parse_network(network_input)
    except Exception as e:
        print(f"输入解析错误: {e}")
        return

    # 2. 配置
    MAX_WORKERS = 50
    TIMEOUT = 3.0

    print(f"开始扫描，共 {len(ip_list)} 个IP，并发 {MAX_WORKERS}...")

    results = []
    # 3. 执行并发任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ip = {executor.submit(scan_ip, ip, TIMEOUT): ip for ip in ip_list}
        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            res = future.result()
            if res:
                print(f"[+] 找到设备: {ip}")
                results.append(res)
            else:
                # 实时反馈每个IP的扫描情况
                print(f"[-] {ip} 无响应")

    # 4. 输出最终汇总
    print(f"\n扫描完成，共发现 {len(results)} 个设备。")
    for res in sorted(results, key=lambda x: ipaddress.ip_address(x['ip'])):
        ip = res.pop('ip')
        print(f"{ip} -> {json.dumps(res, ensure_ascii=False)}")

if __name__ == "__main__":
    main()