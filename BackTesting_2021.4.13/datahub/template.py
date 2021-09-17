#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 数据处理线程 基本
# Author: Tayii
# Data : 2021/1/26
# ----------------------------------------------------
import multiprocessing
from typing import Optional
import pandas as pd

from constant import ALL_BAR_PROCESSED
from datahub.fmt import DataBase
from my_log import ilog
import logging


class DataProcessTemplate(multiprocessing.Process, DataBase):
    """
    数据处理 进程 模板
    """

    def __init__(self,
                 name: str,
                 q_from_caller: multiprocessing.Queue,
                 q_to_caller: multiprocessing.Queue,
                 ):
        multiprocessing.Process.__init__(self, name=name)
        DataBase.__init__(self, name=name)
        self.__q_from_caller = q_to_caller   # 接收来自调用函数数据的queue
        self.__q_to_caller = q_to_caller  # 发送数据给主调函数的queue

        self.__need_to_quit: bool = False  # 需要退出标志
        self.__can_run: bool = True  # 是否能运行

    # @property
    # def ok_bar_slices_done_index(self):
    #     """最后已ok的指针位置"""
    #     return len(self.ok_bar_slices) - 1 if self.ok_bar_slices is not None else -1

    def do_when_all_already_sent(self):
        """所有处理好的数据已经发送完后的操作"""
        self.log(f'所有处理好的数据已经发送完')
        self.send_to_caller(ALL_BAR_PROCESSED, None)  # 发送 所有指标已处理完

    # @property
    # def s_data_doing_index(self) -> int:
    #     return self.__s_data_doing_index
    #
    # def s_data_doing_index_move1(self) -> None:
    #     """往下移1"""
    #     self.__s_data_doing_index += 1
    #
    # @property
    # def has_sent_index(self) -> int:
    #     return self.__ok_bar_slices_waiting_send_index - 1
    #
    # def has_sent_index_move1(self):# todo del
    #     self.__ok_bar_slices_waiting_send_index += 1

    def clear(self) -> None:
        """重置数据和状态位"""
        self.__need_to_quit = False
        self.__can_run = False

    def send_to_caller(self, index, data):
        """发送数据 （index和对应的数据）"""
        self.__q_to_caller.put((index, data))

    @property
    def q_to_caller_is_full(self):
        return self.__q_to_caller.full()

    @property
    def need_quit(self) -> bool:
        """是否需要退出"""
        return self.__need_to_quit

    def quit(self) -> None:
        """退出"""
        self.__need_to_quit = True

    def set_can_run(self) -> None:
        """设置能运行"""
        self.__can_run = True

    def set_stop_run(self) -> None:
        """设置停止运行"""
        self.__can_run = False

    @property
    def can_run(self):
        return self.__can_run

    def log(self, msg, level=logging.DEBUG, exc_info=False) -> None:
        ilog.console(f'{self.name}: {msg}', level=level, exc_info=exc_info)

    def __eq__(self, other):
        if self.name == other.name:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.name)

