#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 数据引擎
# 提供处理好的数据

# Author: Tayii
# Data : 2021/1/12
# ----------------------------------------------------
import time
import datetime
import multiprocessing
from importlib import import_module
from typing import Optional, Any
import pandas as pd
import logging

from backtester.template import BackTestingDataSetting
from dc.config import DataSourceConfig
from dc.source_data import get_source_data_from_config
from datahub.generate import BarGenerator
from datahub.template import DataProcessTemplate
from constant import ALL_BAR_PROCESSED
from strategies.template import StrategyTemplate
from utility import catch_except, timeit


class DataEngine(DataProcessTemplate):
    """
    数据处理引擎
    """

    def __init__(self,
                 q_from_caller: multiprocessing.Queue,  # 接收来自调用函数数据的queue
                 q_to_caller: multiprocessing.Queue,  # 发送数据给调用者的queue
                 data_set_name: str,  # 数据配置 name
                 data_source: DataSourceConfig,  # 数据源配置
                 strategy_import_path: str,  # 策略包完整路径
                 ) -> None:
        DataProcessTemplate.__init__(self, f'{data_set_name} Data Engine',
                                     q_from_caller, q_to_caller, )

        self.ds_name: str = data_set_name  # 配置名
        self.data_source: DataSourceConfig = data_source
        self.strategy_import_path: str = strategy_import_path

        # 原始bar数据
        self.__s_data: Optional[pd.DataFrame] = None
        self.__s_data_doing_index: int = 0  # 原始bar数据 正在处理的位置index
        self.__s_data_interval: int = 0  # 原始数据周期

        # 最终结果，长度跟原始数据一样，计算是基于新bar的每一个切片
        self.__ok_bar_slices: Optional[pd.DataFrame] = None
        self.__ok_bar_slices_waiting_send_index: int = 0  # 即将要发送的数据的指针位置

        self.__indicators: dict = {}  # 需要计算的指标
        self.__new_status: Optional[pd.DataFrame] = None  # 新bar数据 当前状态

        self.__q_to_bg = multiprocessing.Queue(1)  # 跟 bar generate通信
        self.__q_from_bg = multiprocessing.Queue(20)  # 跟 bar generate通信
        self.__indicators: dict = {}

    @property
    def __s_size(self) -> int:
        return len(self.__s_data) if self.__s_data is not None else 0

    @property
    def has_sent_index(self) -> int:
        return self.__ok_bar_slices_waiting_send_index - 1

    @property
    def ok_bar_slices_done_index(self):
        """最后已ok的指针位置"""
        return len(self.__ok_bar_slices) - 1 if self.__ok_bar_slices is not None else -1

    @property
    def __s_data_received_index(self) -> int:
        """新bar数据 最新的已获取切片index"""
        return self.__s_size - 1

    @property
    def __new_status_last_index(self) -> int:
        """当前状态 最新bar index"""
        return self.__new_status.index[-1]  # 新版丢弃了长度，所以不用len() 用index

    def run(self) -> None:
        """
        # 提供统一的切片数据给回测主进程 分发各进程 以提高效率
        """
        self.log(f'进程启动')

        # 新建进程 单独处理 合成bar，提高效率
        bar_gene = BarGenerator(self.__q_to_bg,
                                self.__q_from_bg,
                                self.ds_name,
                                self.data_source,
                                self.strategy_import_path
                                )
        bar_gene.start()

        # 设置指标
        _s = import_module(self.strategy_import_path).Strategy
        strategy: StrategyTemplate = _s(
            name=f'{self.name} strategy',  # 此处支持一次回测多个数据源
            data_source=self.data_source,
        )
        self.__indicators = strategy.data_sets[self.ds_name].indicators
        self.log(f'☆ 加载回测所需计算的指标: {len(self.__indicators)}个'
                 f'\n  {list(self.__indicators.keys())}', logging.INFO)

        if len(self.__indicators) == 0:
            return

        while True:
            if self.need_quit:
                return  # 退出

            if not self.can_run:
                self.log(f'can run == False')
                time.sleep(5)
                continue

            try:
                # 接收主控的指令
                pass

                # 循环处理 receive->process->sent 一次处理一个原始切片
                # 接收新slice 并合并到__new_status里
                if self.__receive_bar_gene_msg():
                    # 对新__new_status 进行处理，生成各指标值
                    self._process_indicators()
                    # 保存 最新切片数据及指标值
                    self.__save_ok_data_slice()

                # 发送
                if self.has_sent_index < self.__s_size - 1:
                    self.bar_data_send_to_caller()
                else:  # 目前收到的数据全部已经处理&发送完
                    if self.__q_from_bg.empty():
                        print(bar_gene, bar_gene.is_alive())
                        if bar_gene is None or not bar_gene.is_alive():
                            # 全部数据都已经收到 退出
                            self.do_when_all_already_sent()
                            return
                        else:
                            # 这种情况 出现在bg来不及处理数据，而自己又已经处理完，所以sleep一下
                            self.log(f'等待 Bar Generate 新数据...')
                            time.sleep(0.5)

                # print(f'在data engine里 s_data=\n', self.__s_data)
                # print(f'处理好的，在data engine里 __new_status=\n', self.__new_status)
                # print(f'输出 在data engine里 ok_bar_slices=\n', self.__ok_bar_slices.tail(10))

                # 辅助显示
                if self.has_sent_index % 53 == 33:
                    self.log(f'ok_bar {self.has_sent_index} / generated_bar {self.__s_size}')

            except Exception as e:
                self.log(f'{e}', logging.ERROR, exc_info=True)
                time.sleep(0.5)

    def bar_data_send_to_caller(self) -> None:
        """
        发送已处理好的bar数据 给主控
        """
        if self.q_to_caller_is_full:
            return

        if self.has_sent_index < self.ok_bar_slices_done_index:
            # 有可以发送的数据
            self.send_to_caller(self.__ok_bar_slices_waiting_send_index,
                                self.__ok_bar_slices.iloc[self.__ok_bar_slices_waiting_send_index])
            self.__ok_bar_slices_waiting_send_index += 1

    # @timeit
    @catch_except(ret=False)
    def __receive_bar_gene_msg(self,
                               drop_len: int = 1000,  # 丢弃的长度 前面的没用了，放着太浪费时间
                               ) -> bool:
        """
        获取 bar generate 生成的新bar切片 / 或者 结束消息
        Returns:
            True: 收到新切片
            False: 没有收到 或者异常
        """
        if self.__q_from_bg.empty():
            if self.__s_data_received_index == self.has_sent_index:

                time.sleep(0.5)

            self.log(f'self.__s_data_received_index{self.__s_data_received_index} '
                     f' self.has_sent_index{self.has_sent_index}')
            return False

        index, new_slice = self.__q_from_bg.get()

        if index == ALL_BAR_PROCESSED:  # 收到结束标记
            self.log(f'Bar Generate 所有bar已合成完。')
            self.__bar_gen = None  # 线程退出 由bg自己处理
            return False  # 退出

        # print(f'received {index} slice;')
        if self.__s_size == 0:
            self.__s_data = pd.DataFrame(new_slice).T
            self.__new_status = self.__s_data.copy()
        else:
            self.__s_data = self.__s_data.append(new_slice, ignore_index=True)
            # 判断 new_status是否需要创建新的bar
            new_index: int = new_slice['new_index']
            if self.__new_status.index[-1] == new_index:
                for k in new_slice.index:
                    self.__new_status.loc[new_index, k] = new_slice[k]  # 覆盖
            else:
                self.__new_status.loc[new_index] = new_slice

            if len(self.__new_status) > 2 * drop_len:
                self.__new_status = self.__new_status[drop_len-2:]  # 把前面用不着了的丢掉

        return True

    def __save_ok_data_slice(self) -> None:
        """
        把处理好的当前状态的切片（数据和指标） 保存
        """
        if self.__ok_bar_slices is None:
            self.__ok_bar_slices = self.__new_status.copy()
        else:
            s = self.__new_status.loc[self.__new_status_last_index]
            self.__ok_bar_slices = self.__ok_bar_slices.append(s, ignore_index=True)

    def _process_indicators(self) -> None:
        """
        按单bar计算各指标
        每次只处理一个bar数据
        """
        # 使用新bar最新状态数据（截止当前）计算  # [['Open', 'High', 'Low', 'Last', 'Volume', ...]]
        # 计算各指标 值放入self.__new_bars，保证指标值在新bar当前状态下正常
        for indicator, handle in self.__indicators.items():
            # _df = self.__new_status[:]
            r: Any = handle(self.__new_status)  # todo 前面可以考虑再截短

            if r is None:
                continue  # 前面几个bar 有时候可能没法生成数据

            # 一系列指标 只更新current bar
            if isinstance(r, pd.Series):
                for ind, v in r.items():
                    self.__new_status.loc[self.__new_status_last_index, ind] = v
            # 一系列指标，且对多bar(index)操作 其中一个是self.__new_bars_index
            elif isinstance(r, list):
                for s in r:
                    if isinstance(s, pd.Series):
                        change_bar_index = s['change_bar_index']
                        s_ = s.copy().drop('change_bar_index')
                        for ind in s_.index:
                            self.__new_status.loc[change_bar_index, ind] = s_[ind]
            # 一系列指标
            elif isinstance(r, dict):
                for ind in r.keys():
                    self.__new_status.loc[self.__new_status_last_index, ind] = r[ind]
            # 单一指标
            else:
                self.__new_status.loc[self.__new_status_last_index, indicator] = r
