#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 交易流程 主控处理
# Author: Tayii
# Data : 2020/12/02
# ----------------------------------------------------
import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional, List
import time
import datetime
from logging import INFO
import threading
from multiprocessing import Queue, Manager, Process, Pool, cpu_count
from abc import ABC, abstractmethod
import pandas as pd
from importlib import import_module
from os import path
from dataclasses import asdict

from backtester.template import BackTestingDataSetting
from config import BACK_TESTING_RESULT_DIR
from datahub.indicator import IndicatorCalc
from dc.config import DataSourceConfig, YMH21_1_SC, YM_2019year1_SC, ESH21_1_SC, B6H21_1_SC
from my_log import ilog
from strategies.template import StrategyTemplate

from .data import DataEngine

from constant import Sec, ALL_BAR_PROCESSED
from utility import timeit, catch_except
from backtester.back_testing_run import BackTester


# ---------------------------------
# 主控流程 基类
# ---------------------------------
class BaseProcess(Process):
    def __init__(self, q: Queue, name: str = None):
        """
        Args:
            q: Queue 前后台通信 及 主控流程内各模块通信
        """
        Process.__init__(self)
        self.q = q  # 跟主进程（UI）通信
        self.name = name or 'Base Process'

    @abstractmethod
    def run(self) -> None:
        pass

    def log(self, msg, level=logging.DEBUG, exc_info=False) -> None:
        ilog.console(f'{self.name}: {msg}', level=level, exc_info=exc_info)


class BackTestingProcess2(BaseProcess):
    """
    回测 主控流程 202104 new
    """

    def __init__(self, q: Queue):
        BaseProcess.__init__(self, q)
        self.name = 'Back-Testing Process 202104 new'

    @timeit
    def run(self) -> None:
        """"""
        self.log(f'进程启动', logging.INFO)


# ---------------------------------
# 回测 主控流程
# ---------------------------------
class BackTestingProcess(BaseProcess):
    def __init__(self, q: Queue):
        BaseProcess.__init__(self, q)
        self.name = 'Back Testing Process'

        # 数据处理模块
        self.data_engine = None
        # to数据处理模块（进程）的queue
        self.__q_to_de: Dict[str, multiprocessing.Queue] = {}
        # from数据处理模块（进程）的queue
        self.__q_from_de: Dict[str, multiprocessing.Queue] = {}
        # 接收器 线程，接收DE数据
        self.receiver: Dict[str, threading.Thread] = {}

        self.__ok_data: Dict[str, pd.DataFrame] = {}

        self.start_time = None  # 测试开始时间

    @timeit
    def run(self) -> None:
        """"""
        self.log(f'进程启动', logging.INFO)

        # 模拟 发一条 回测数据处理命令 ---
        # 选择的策略 文件名
        # ## 此处运行 ==============================================
        data_source_list = [YMH21_1_SC, ]  # YM_2019year1_SC  ESH21_1_SC B6H21_1_SC
        strategy_import_path = 'strategies.strategy_hour1_pinbar_0412'
        for data_source in data_source_list:
            self.execute_a_back_testing_command(
                name='strategy_hour1_pinbar_0413',
                data_source=data_source,
                strategy_import_path=strategy_import_path,
                description=f'60分钟 pin-bar',
                start_id=0,
            )  # todo 由UI控制

        while True:
            # TODO 接收外部调用的命令

            self.log(f'no task, waiting......')

            # show_active_threads()
            time.sleep(60)

    def execute_a_back_testing_command(self,
                                       name: str,  # 回测计划名==存放文件夹名
                                       data_source: DataSourceConfig,  # 数据源配置
                                       strategy_import_path: str,  # 策略包完整路径
                                       description: str = ' ',
                                       start_id: int = 0,  # 从第几组开始（断点续传）
                                       ) -> None:
        """执行一条 回测数据处理命令"""

        # 生成策略实例
        _s = import_module(strategy_import_path).Strategy
        strategy: StrategyTemplate = _s(
            data_source=data_source,
            name=f'{name}_{data_source.symbol}',  # 此处支持一次回测多个数据源
        )
        strategy.import_path = strategy_import_path

        self.log(f'开始一个新的回测计划：{strategy.name} [{description}]')

        # 本次回测 所有数据存放的文件夹
        bt_dir = path.join(BACK_TESTING_RESULT_DIR, f'{strategy.name}')
        if not path.exists(bt_dir):
            os.makedirs(bt_dir)

        # 记录 回测总体信息
        _a_data_set: BackTestingDataSetting = list(strategy.data_sets.values())[0]
        outline = {
            'description': description,  # 描述
            'symbol': _a_data_set.symbol,
            'hyper_parameter_name': list(strategy.hyper_parameter[0].keys()),  # 超参数名
        }
        outline.update(asdict(strategy.trade_params))
        outline['open_max'] = outline['open_max'].name
        outline_filepath = path.join(bt_dir, f'outline_filepath.txt')
        with open(outline_filepath, 'w') as f:
            f.write(json.dumps(outline))
            self.log(f'outline写入txt')

        # 数据源 多周期数据及指标处理
        self.start_time = time.time()  # 数据处理起始时间
        for ds_name, data_set in strategy.data_sets.items():
            # 保存的数据文件地址
            ok_data_filepath = path.join(bt_dir, f'{ds_name}.csv')

            # 获取处理好的数据 或调用Data Engine处理
            if path.exists(ok_data_filepath):
                # 之前有保存，就直接用 （需要人为判断，这个数据是否是跟策略需要的一样的）
                self.log(f'{ds_name}有原先处理好的，直接读取使用')
                df_ = pd.read_csv(ok_data_filepath)[3333:33333]

                # 对时间进行转换
                df_['datetime'] = df_['datetime'].apply(
                    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                # todo 可能还有其他要转换
                self.__ok_data[ds_name] = df_
            else:
                self.__use_de_to_calc(ds_name,
                                      data_set.data_source,
                                      strategy_import_path,
                                      strategy.save_data_result,
                                      ok_data_filepath)

        # 等待接收完
        def _check():
            """检查数据是否全部处理完"""
            self.log('receiving...................')
            for ds_name, thrend_ in self.receiver.items():
                if thrend_ and thrend_.is_alive():
                    return False
            return True  # 接收线程全部结束

        while True:
            if _check():
                break
            time.sleep(5)

        # 处理一次性计算指标  # todo del
        # for ds_name, _ in strategy.data_sets.items():
        #     # 加 avg_range_ratio
        #     avg_range_ratio: pd.Series = IndicatorCalc.days_avg_range(data=self.__ok_data[ds_name])
        #     self.__ok_data[ds_name]['avg_range_ratio'] = self.__ok_data[ds_name].apply(
        #         lambda x: avg_range_ratio[x.m_Date], axis=1)

        self.log(f'所有需要的数据都已经准备好了，总用时[{time.time() - self.start_time:.1f}]秒')
        if strategy.show_data_result:
            self.log(f'准备好的数据如下：')
            for k, v in self.__ok_data.items():
                print(f'{k}:\n', v)

        # 开启多进程 进行回测
        self.start_time = time.time()  # 回测开始计时
        self.__multiprocessing_back_testing(strategy, data_source, start_id=start_id)

        self.log(f'多进程回测，总用时[{time.time() - self.start_time:.1f}]秒')
        self.start_time = None

        # 处理并显示回测结果
        # 读取结果文件 总+分

    @catch_except(ret=False)
    def __use_de_to_calc(self,
                         ds_name: str,  # data_set name
                         data_source: DataSourceConfig,  # 数据源配置
                         strategy_import_path: str,  # 策略包完整路径
                         save_data_result: bool,  # 是否保存数据
                         ok_data_filepath: str = ''  # 保存数据的地址
                         ) -> None:
        """
        根据策略 获取新的基础数据 并进行处理（获取需要的指标值）
        """
        self.__ok_data[ds_name] = None  # 清空之前的

        # 每次新处理回测数据都新建进程（防止出错），结束后释放（节约资源）
        if self.data_engine:
            self.data_engine.quit()
        # 新开启数据处理引擎(计算额外指标等)
        self.__q_to_de[ds_name] = multiprocessing.Queue(1)
        self.__q_from_de[ds_name] = multiprocessing.Queue(10)
        self.data_engine = DataEngine(self.__q_to_de[ds_name],
                                      self.__q_from_de[ds_name],
                                      ds_name,
                                      data_source,
                                      strategy_import_path, )
        self.data_engine.start()

        self.log(f'{ds_name}开启receiver线程 用于获取data engine处理好的数据')
        self.receiver[ds_name] = threading.Thread(
            target=self.__receive_from_data_engine,
            args=(ds_name, save_data_result, ok_data_filepath),
            name=f"{ds_name} DE data receiver", )
        self.receiver[ds_name].daemon = True
        self.receiver[ds_name].start()

    def __multiprocessing_back_testing(self,
                                       strategy: StrategyTemplate,
                                       data_source: DataSourceConfig,  # 数据源配置
                                       start_id: int = 0,  # 从第几组开始（断点续传）
                                       use_cpus: int = None,  # 使用几个核  None默认
                                       ) -> None:
        """多进程 回测"""

        total_result_filepath = path.join(BACK_TESTING_RESULT_DIR,
                                          f'{strategy.name}',
                                          f'total_result.txt')

        def save_call_back(msg):
            """保存 回测进程 结果"""
            if msg:
                with open(total_result_filepath, 'a+')as f:
                    line = str(msg) + "\n"
                    f.write(line)

        size = max(len(strategy.hyper_parameter), 1)  # 没有超参数（完全靠逻辑） ==1组
        self.log(f'{strategy.name} 需要回测[ {size} ]组参数')

        # Global = Manager().Namespace()
        # Global.ok_data = self.__ok_data
        _cpus = min(use_cpus, cpu_count()) if use_cpus else cpu_count() - 1
        self.log(f'☆ 开启多进程(调用{_cpus}核) 进行回测...')
        with Pool(processes=_cpus, ) as pool:
            # TODO 此处可以考虑 全部参数放这里，或者部分放子进程里

            for i in range(start_id, size):
                time.sleep(2)
                pool.apply_async(BackTester.back_testing_run,
                                 args=(strategy.name,
                                       data_source,  # 数据源配置
                                       strategy.import_path,  # 选择的策略 包完整路径
                                       self.__ok_data,  # Global.ok_data,  # 处理好的数据 全部
                                       i,  # 回测 第n组超参数
                                       strategy.save_trade_result,  # 是否保存回测交易结果
                                       ),
                                 callback=save_call_back,
                                 )
            pool.close()
            pool.join()

        self.log(f'多进程 全部进程结束。。。')

    def __receive_from_data_engine(self,
                                   ds_name: str,  # data_set名
                                   save_data_result: bool,  # 是否保存数据
                                   ok_data_filepath: str = ''  # 保存数据的地址
                                   ) -> None:
        """接收Data Engine传送来已处理好的数据"""
        df: Optional[pd.DataFrame] = None
        q: Queue = self.__q_from_de[ds_name]
        slice: int = 1

        while True:
            # time.sleep(0.01)
            try:
                index, data = q.get(block=True, timeout=None)  # 一直阻塞

                if index == ALL_BAR_PROCESSED:
                    self.data_engine.quit()  # 收到结束标记
                    self.__ok_data[ds_name] = df  # 保存
                    self.log(f'{ds_name} 已收到 Data Engine 处理好的所有带指标新bar数据。'
                             f'用时[{time.time() - self.start_time:.1f}]秒')
                    print(df)
                    if save_data_result and df is not None:
                        # 全部实时数据和指标值都已经接受到
                        # 处理可以统一计算的指标 # todo
                        # self.__process_once_indicators()
                        self.__ok_data[ds_name].to_csv(ok_data_filepath)
                        self.log(f'DE已处理完{ds_name}的指标数据 并保存到{ok_data_filepath}')

                    return  # 直接退出

                # 拼接数据
                if df is None:
                    df = data.to_frame()
                    if len(df.columns) == 1:
                        df = df.T  # 应该需要行列互换一下 pd比较傻
                else:
                    df.loc[index] = data
                    max_per_file = 20000
                    if len(df) > max_per_file:
                        name, suffix = ok_data_filepath.split('.')
                        df[:max_per_file].to_csv(f'{name}_{slice}.{suffix}')
                        df = df[max_per_file:]
                        slice += 1

            except Exception as e:
                self.log(f'{ds_name} receive from data engine err: {e}', logging.ERROR)

    def __process_once_indicators(self,
                                  ds_name: str,  # data_set名
                                  ):
        """处理可以统一计算的指标"""


# ---------------------------------
# 交易 主控流程
# ---------------------------------
class TradingProcess(BaseProcess):

    def __init__(self, q: Queue):
        BaseProcess.__init__(self, q)
        self.name = 'Trading Process'

    def run(self) -> None:
        self.log(f'{self.name} 线程启动')
        while True:
            time.sleep(10)

    def init_additional_engines(self) -> None:
        """
        Init addtional engines.
        """
        pass
