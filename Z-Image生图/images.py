#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图片生成脚本 - 读取JSON配置，发送请求，保存返回的图片
自动创建时间戳文件夹，保存图片和配置文件
"""

import json
import base64
import requests
import os
import shutil
from datetime import datetime
from pathlib import Path

REQUEST_TIMEOUT_SECONDS = int(os.getenv("Z_IMAGE_TIMEOUT_SECONDS", "1200"))

def load_config(config_file="config.json"):
    """
    从JSON文件加载配置

    Args:
        config_file: 配置文件路径

    Returns:
        dict: 配置字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ 成功加载配置文件: {config_file}")
        return config
    except FileNotFoundError:
        print(f"✗ 错误: 找不到配置文件 {config_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"✗ 错误: JSON格式无效 - {e}")
        return None

def send_generation_request(config, api_url, api_key=None):
    """
    发送图片生成请求

    Args:
        config: 生成配置字典
        api_url: API端点URL
        api_key: API密钥（可选）

    Returns:
        dict: API返回的JSON数据
    """
    headers = {"Content-Type": "application/json"}

    # 如果提供了API密钥，添加到请求头
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        print(f"→ 正在发送请求到: {api_url}")
        print(f"→ 使用配置: {json.dumps(config, ensure_ascii=False, indent=2)}")

        response = requests.post(
            api_url,
            json=config,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS
        )

        # 检查HTTP状态
        response.raise_for_status()

        # 解析JSON响应
        result = response.json()
        print(f"✓ 请求成功 (状态码: {response.status_code})")
        return result

    except requests.exceptions.Timeout:
        print(f"✗ 错误: 请求超过 {REQUEST_TIMEOUT_SECONDS} 秒仍未完成，请检查服务日志、GPU 状态或降低分辨率/步数")
        return None
    except requests.exceptions.ConnectionError:
        print(f"✗ 错误: 无法连接到服务器，请检查URL")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP错误: {e}")
        if 'response' in locals() and response.text:
            print(f"响应内容: {response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"✗ 错误: 无法解析JSON响应 - {e}")
        if 'response' in locals():
            print(f"原始响应: {response.text[:200]}...")
        return None
    except Exception as e:
        print(f"✗ 未知错误: {e}")
        return None

def create_output_folder():
    """
    创建以当前时间命名的输出文件夹

    Returns:
        Path: 创建的文件夹路径
    """
    # 格式: 20260824_153045
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"生成_{timestamp}"

    # 在当前目录创建文件夹
    folder_path = Path.cwd() / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    print(f"✓ 创建输出文件夹: {folder_path.absolute()}")
    return folder_path

def parse_and_save_image(response_data, output_dir, config_file="config.json"):
    """
    解析响应数据中的base64图片并保存，同时复制配置文件

    Args:
        response_data: API返回的JSON数据
        output_dir: 输出目录路径
        config_file: 配置文件路径

    Returns:
        str: 保存的图片路径，失败返回None
    """
    try:
        # 检查数据结构
        if not response_data or "data" not in response_data:
            print(f"✗ 错误: 响应中没有 'data' 字段")
            print(f"响应内容: {json.dumps(response_data, ensure_ascii=False, indent=2)[:500]}")
            return None

        data_list = response_data["data"]
        if not data_list or len(data_list) == 0:
            print(f"✗ 错误: 'data' 数组为空")
            return None

        # 提取base64数据
        first_item = data_list[0]
        if "b64_json" not in first_item:
            print(f"✗ 错误: 数据项中没有 'b64_json' 字段")
            print(f"可用字段: {list(first_item.keys())}")
            return None

        b64_data = first_item["b64_json"]
        if not b64_data:
            print(f"✗ 错误: base64数据为空")
            return None

        # 解码base64
        try:
            # 如果数据包含前缀 (如 "data:image/png;base64,")，去掉它
            if "," in b64_data and b64_data.startswith("data:"):
                b64_data = b64_data.split(",", 1)[1]

            image_bytes = base64.b64decode(b64_data)
            print(f"✓ 成功解码base64数据 ({len(image_bytes)} 字节)")
        except Exception as e:
            print(f"✗ 错误: base64解码失败 - {e}")
            return None

        # 获取模型名称用于文件名
        model_name = response_data.get("model", "unknown_model")

        # 生成图片文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"image_{model_name}_{timestamp}.png"
        image_path = output_dir / image_filename

        # 保存图片
        try:
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            print(f"✓ 图片已保存: {image_path.absolute()}")
        except Exception as e:
            print(f"✗ 错误: 保存图片失败 - {e}")
            return None

        # 复制配置文件到输出文件夹
        try:
            config_src = Path(config_file)
            if config_src.exists():
                config_dst = output_dir / config_src.name
                shutil.copy2(config_src, config_dst)
                print(f"✓ 配置文件已复制: {config_dst.absolute()}")
            else:
                print(f"⚠ 警告: 配置文件不存在，跳过复制")
        except Exception as e:
            print(f"⚠ 警告: 复制配置文件失败 - {e}")

        # 保存响应信息到JSON文件（便于查看完整响应）
        try:
            response_info = {
                "model": response_data.get("model", "unknown"),
                "generated_at": datetime.now().isoformat(),
                "image_file": image_filename,
                "config_used": config
            }
            info_path = output_dir / "response_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(response_info, f, ensure_ascii=False, indent=2)
            print(f"✓ 响应信息已保存: {info_path.absolute()}")
        except Exception as e:
            print(f"⚠ 警告: 保存响应信息失败 - {e}")

        return str(image_path.absolute())

    except Exception as e:
        print(f"✗ 错误: 处理响应数据时出错 - {e}")
        return None

def main():
    """
    主函数
    """
    # ===== 配置区域 =====
    CONFIG_FILE = "config.json"  # 配置文件路径
    API_URL = "http://192.168.4.137:18001/v1/images"  # 替换为实际的API地址
    API_KEY = "zw79LaXQigQxIu9/mBJb8iplzx/Lx5tim2Q6JK0B/Os="  # 如果需要认证，在这里设置

    # ===== 执行流程 =====
    print("=" * 60)
    print("图片生成脚本")
    print("=" * 60)

    # 1. 加载配置
    config = load_config(CONFIG_FILE)
    if not config:
        print("✗ 脚本终止")
        return

    # 2. 创建输出文件夹（时间戳命名）
    output_folder = create_output_folder()

    # 3. 发送请求
    response = send_generation_request(config, API_URL, API_KEY)
    if not response:
        print("✗ 脚本终止")
        return

    # 4. 解析并保存图片
    print("\n" + "=" * 60)
    print("处理响应数据")
    print("=" * 60)

    saved_path = parse_and_save_image(response, output_folder, CONFIG_FILE)

    # 5. 输出结果
    print("\n" + "=" * 60)
    if saved_path:
        print(f"✓ 成功! 所有文件已保存到文件夹:")
        print(f"  文件夹: {output_folder.absolute()}")
        print(f"  ├── {Path(saved_path).name}")
        print(f"  ├── {Path(CONFIG_FILE).name}")
        print(f"  └── response_info.json")
        print(f"\n✓ 模型: {response.get('model', 'unknown')}")
        if "data" in response and len(response["data"]) > 0:
            data_item = response["data"][0]
            for key in data_item:
                if key != "b64_json":
                    print(f"✓ 额外信息 - {key}: {data_item[key]}")
    else:
        print("✗ 图片保存失败，请检查响应格式")

    print("=" * 60)

if __name__ == "__main__":
    main()
