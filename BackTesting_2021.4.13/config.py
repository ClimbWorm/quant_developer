#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
配置文件
"""
import logging
from os import path, mkdir, makedirs
from logging import INFO, DEBUG,CRITICAL
from typing import Dict, Any
# from tzlocal import get_localzone


# 开关 （加了前端后可以放前端去）
LOCAL_DEBUGGING = True,  # 运行环境，是否本地测试环境,  True:是  False:生成环境
ENABLE_BACK_TESTING = True  # 是否进行 回测
ENABLE_TRADING = False  # 是否进行 交易（实盘）


# 获取当前文件路径
CURRENT_PATH = path.abspath(__file__)
ROOT_DIR = path.dirname(CURRENT_PATH)  # 在本地类似'E:\\workspace\\DigitalCurrency'

# 回测数据结果目录
BACK_TESTING_RESULT_DIR = path.join(ROOT_DIR, 'back_testing_result')
if not path.exists(BACK_TESTING_RESULT_DIR):
    makedirs(BACK_TESTING_RESULT_DIR)  # makedirs可创建多级目录

BACK_TESTING_SOURCE_DATA_DIR = path.join(ROOT_DIR, 'source_data')
if not path.exists(BACK_TESTING_SOURCE_DATA_DIR):
    makedirs(BACK_TESTING_SOURCE_DATA_DIR)

# 日志相关 -------------------------
_logs_path = path.join(ROOT_DIR, 'logs')  # 日志存放路径
if not path.exists(_logs_path):
    mkdir(_logs_path)  # 如果不存在这个logs文件夹，就自动创建一个

# 日志级别
LOG_SETTING = {
    'file_dir': _logs_path,  # 日志存放路径
    'level': {
        'main': logging.DEBUG,  # root日志级别
        'console': logging.DEBUG if LOCAL_DEBUGGING else logging.ERROR,  # 输出到控制台的日志级别
        'file': logging.INFO,  # 保存到文件的日志级别
        'ui': logging.INFO,  # 输出到前端的日志级别
        },
    }
