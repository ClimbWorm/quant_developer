#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 程序 主入口
# Author: Tayii
# Data : 2020/12/01
# ----------------------------------------------------
import time
import sys
# from queue import Queue
from multiprocessing import Queue

from engine import BackTestingProcess, TradingProcess
from config import ENABLE_BACK_TESTING, ENABLE_TRADING
from constant import MProcess

from engine import *
import logging

from engine.process import BackTestingProcess2


def main():
    """主流程"""

    print(f'  - 系统启动 -')

    # 加载 UI
    ############################

    bt_queue = Queue()  # 回测主流程用
    trade_queue = Queue()  # 交易主流程用

    bt_process = None
    trade_process = None

    if ENABLE_BACK_TESTING and bt_process is None:
        bt_process = BackTestingProcess(bt_queue)
        bt_process.start()

    if ENABLE_TRADING and trade_process is None:
        trade_process = TradingProcess(trade_queue)
        trade_process.start()

    # 启动 UI
    ############################

    need_exit = False
    while True:
        time.sleep(5)

        if need_exit:
            try:
                sys.exit(0)
            except Exception as e:
                print(f'系统退出 {e}')


if __name__ == '__main__':
    main()
