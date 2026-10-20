print("Hello")


# 1. 打印Python版本（和Java的System.getProperty("java.version")类似）
import sys
print(sys.version)

# 2. 简单运算
print(10 + 20)


print(0b100)  # 二进制整数
print(0o100)  # 八进制整数
print(100)    # 十进制整数
print(0x100)  # 十六进制整数


"""
使用变量保存数据并进行加减乘除运算

Version: 1.0
Author: 骆昊
"""
a = 45        # 定义变量a，赋值45
b = 12        # 定义变量b，赋值12
print(a, b)   # 45 12
print(a + b)  # 57
print(a - b)  # 33
print(a * b)  # 540
print(a / b)  # 3.75