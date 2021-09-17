#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 准备数据源
#
# Author: Tayii
# Data : 2021/4/4
# ----------------------------------------------------
import logging
import multiprocessing
import random
import time
from multiprocessing import Queue, cpu_count, Pool
from dataclasses import dataclass, asdict
import datetime
from typing import Optional, Any
from os import path

import pandas as pd

from config import BACK_TESTING_RESULT_DIR, BACK_TESTING_SOURCE_DATA_DIR
from constant import Sec
from datahub.template import DataProcessTemplate
from dc.config import DataSourceConfig, SC_BAR_DATA_Columns
from dc.sc import get_bar_data_from_txt
from dc.source_data import get_source_data_from_config
from engine.process import BaseProcess
from utility import catch_except


def by_tick():
    """
    由tick数据生成
    """
    pass


def by_bar(start_index: int,  # 起始计算的index
           end_index: int,  # 终止计算的index
           source_df: pd.DataFrame,  # 计算用的原始数据
           **kwargs):
    """
    由bars数据生成 小周期->大周期
    """
    source_bar_interval: int = kwargs['source_bar_interval']
    new_bar_interval: int = kwargs['new_bar_interval']
    new_bar_day_open = kwargs.get(  # 新bar 日盘开盘时间
        'new_bar_day_open', datetime.time(8))
    new_bar_night_open = kwargs.get(  # 新bar 夜盘开盘时间
        'new_bar_night_open', )

    prior_bar_dt: Optional[datetime.datetime] = None
    ok_bars: Optional[pd.DataFrame] = None
    ok_bars_index: Optional[int] = None

    def _find_nearest_dt(start_: datetime.datetime, end: datetime.datetime):
        """找到此bar属于的进入时间"""
        m = (end - start_).total_seconds() // new_bar_interval
        return start_ + datetime.timedelta(seconds=new_bar_interval * m)

    curr_index = max(0, start_index)
    end_i = min(source_df.index[-1], end_index)
    print(f'by_bar {curr_index} - {end_i} 进程启动')

    size = end_i - start_index
    _mark_size: int = int(size / random.uniform(4.5, 9))  # 显示用
    begin = time.time()

    while curr_index <= end_i:

        if curr_index % _mark_size == 0:
            print(f'处理了{curr_index / size:.1%} 耗时：{time.time() - begin:.1f}秒')

        # 当前要处理 原始数据的bar
        curr_bar = source_df.iloc[curr_index].copy()
        curr_bar_dt: datetime.datetime = curr_bar['datetime']  # 这条数据的开始时间
        _interval = datetime.timedelta(seconds=new_bar_interval)

        def __get_dt_by_time(dt: datetime.datetime, days: int, time_: datetime.time):
            # 设置开始时间（供第1bar检索起始用）
            return (dt - datetime.timedelta(days=days)).replace(
                hour=time_.hour,
                minute=time_.minute,
                second=time_.second,
                microsecond=0)

        def _new_bar(prior_dt: datetime.datetime) -> None:
            """
            目标周期 添加新bar
            Args:
                prior_dt: 目标周期上个bar 开始的datetime
            """
            nonlocal prior_bar_dt, ok_bars, curr_index, ok_bars_index

            ok_bars_index += 1
            _bar_dt = _find_nearest_dt(prior_dt, curr_bar_dt)  # 进入时间

            curr_bar['datetime'] = _bar_dt  # 此bar开始时间
            curr_bar['new_index'] = curr_index
            if ok_bars is None:
                ok_bars = pd.DataFrame(curr_bar).T
            else:
                ok_bars = ok_bars.append(curr_bar, ignore_index=True)

            prior_bar_dt = _bar_dt
            curr_index += 1

        # 主逻辑 ===================
        curr_day_open = __get_dt_by_time(curr_bar_dt, 0, new_bar_day_open)
        curr_night_open = (__get_dt_by_time(curr_bar_dt, 0, new_bar_night_open)
                        if new_bar_night_open else None)

        if curr_index == start_index:  # 第一个 刚开始处理
            if new_bar_night_open and curr_bar_dt.time() < new_bar_day_open:
                # 从前一日夜盘开盘 开始
                start = __get_dt_by_time(curr_bar_dt, 1, new_bar_day_open)
            elif new_bar_night_open is None or curr_bar_dt.time() < new_bar_night_open:
                start = curr_day_open  # 日盘开盘 开始
            else:
                start = curr_night_open  # 夜盘开盘 开始

            _new_bar(start)  # 新bar
            continue

        # 如果正跨过开盘时间，直接设置为新bar
        if prior_bar_dt < curr_day_open <= curr_bar_dt:
            _new_bar(prior_dt=curr_day_open)  # 新bar
            continue
        elif new_bar_night_open and prior_bar_dt < curr_night_open <= curr_bar_dt:
            _new_bar(prior_dt=curr_night_open)  # 新bar
            continue

        # 如果超过周期时间
        if (curr_bar_dt - prior_bar_dt).total_seconds() >= new_bar_interval:
            _new_bar(prior_dt=prior_bar_dt)  # 基于前bar->新bar
            continue

        # 都不是，则是一个大周期内，在前bar基础上拼接 --------------------
        bar_: pd.Series = ok_bars.iloc[curr_index-1].copy()
        for item in bar_.keys():
            # 必选 OHLC  open不动
            if item in ['High', 'high']:
                bar_[item] = max(bar_[item], curr_bar[item])
            elif item in ['Low', 'low']:
                bar_[item] = min(bar_[item], curr_bar[item])
            elif item in ['Close', 'close', 'Last', 'last']:
                bar_[item] = curr_bar[item]
            elif item in ['Vol', 'vol', 'Volume', 'volume']:
                bar_[item] += curr_bar[item]
            # 日期之类 new_index 不变
            # 其他 todo

        # 保存新bar切片
        ok_bars = ok_bars.append(bar_, ignore_index=True)
        curr_index += 1

    print(f'by_bar {curr_index} - {end_i} 处理完')
    print(ok_bars)
    return ok_bars

class BarGenerator2(BaseProcess):
    """
    拼接 由小周期生成大周期的新bar 202104 new
    """

    def __init__(self,
                 q: multiprocessing.Queue,  # queue
                 f_name: str,  # 生成文件名
                 f_source_name: str,  # 源文件名
                 **kwargs,
                 ):
        """Constructor"""

        BaseProcess.__init__(self, q, f'Bar Generate')
        self.f_name: str = f_name  # 生成文件名
        self.f_source_name: str = f_source_name
        self.source_bar_interval: int = kwargs.get('source_bar_interval', )  # 源bar的周期 秒
        self.day_open = kwargs.get('day_open', 8.5)  # 每天开盘的时间 小时计 （可选）
        self.need_columns = kwargs.get('need_columns',  # 策略需要的数据（列名）
                                       ['datetime', 'Open', 'High', 'Low', 'Last',
                                        'Volume', 'timestamp'])
        self.ds_from = kwargs.get('ds_from', 'sc')  # 回测数据来源
        self.kwargs = kwargs

        self.df_source: pd.DataFrame = None

    def run(self) -> None:
        """运行"""
        self.log(f'进程启动')

        # 读取原始数据
        df = self.__get_source_data()
        if df is None or len(df) == 0:
            self.log(f'__get_source_data err', logging.ERROR)
            raise Exception(f'__get_source_data err')
        else:
            self.df_source = df  # todo del

        self.log(f'☆ 加载回测原始数据 共{df}条...')

        # 决定由tick生成还是bar生成新bar
        generate = (by_bar if self.source_bar_interval > Sec.TICK.value else by_tick)

        self.bar_generate(generate=generate, use_cpus=1, )

    @catch_except()
    def __get_source_data(self,
                          size: float = 1e7,
                          ) -> Optional[pd.DataFrame]:
        """
        获取要回测的原始数据 OHLC等 及 配置参数
        """
        try:
            if self.ds_from == 'sc':
                df: pd.DataFrame = get_bar_data_from_txt(
                    filepath=path.join(BACK_TESTING_SOURCE_DATA_DIR, self.f_source_name),
                    size=size,
                    columns=SC_BAR_DATA_Columns,
                )
            else:
                raise NotImplementedError()
        except Exception as e:
            print(f'__get_source_data {e}')
            return
        # else:
        #     # TODO 格式等判断

        if df is None or df.index[0] != 0:  # index要0开头 升序
            self.log(f'df.index[0] != 0', logging.ERROR)
            return

        # 是否只需要某些列
        if self.need_columns is not None:
            if not set(self.need_columns).issubset(df.columns):
                self.log(f"{self.need_columns} not in df.columns", logging.ERROR)
                return
            else:
                df = df[self.need_columns]

        # 加一列，偏移开盘时间后 属于哪天 ---------
        t_offset = self.day_open * 60 * 60

        def f(t):
            new_t = t - t_offset
            return datetime.datetime.fromtimestamp(new_t).date()

        df['m_Date'] = df['timestamp'].apply(f)

        return df

    def bar_generate(self,
                     generate: Any,  # 处理接口
                     start_id: int = 0,  # 从第几组开始（断点续传）
                     use_before_index=0,  # 需要前面多少根bar数据支援
                     use_cpus: int = None,  # 使用几个核  None默认
                     ):
        """
        拼接bar
        """
        # 多进程一起跑
        _cpus = min(use_cpus, cpu_count()) if use_cpus else cpu_count() - 1
        chunk: int = len(self.df_source) // _cpus + 1  # 每个进程处理的数据
        self.log(f'☆ 开启多进程(调用{_cpus}核) 进行回测...')
        with Pool(processes=_cpus, ) as pool:
            for i in range(start_id, _cpus):
                pool.apply_async(generate,
                                 args=(i * chunk, (i + 1) * chunk,
                                       self.df_source[i * chunk - use_before_index:(i + 1) * chunk]),
                                 kwds=self.kwargs,
                                 # callback=save_call_back,
                                 )
            pool.close()
            pool.join()

        self.log(f'多进程 全部进程结束。。。')


def merge_csv_files(strategy_name: str,  #
                    file_name: str,
                    n=1,
                    need_save=False):
    """拼接不同的csv成一整个"""
    file_dir = path.join(BACK_TESTING_RESULT_DIR, strategy_name)
    file_path = path.join(file_dir, file_name)
    name, suffix = file_name.split('.')
    df = pd.DataFrame()

    for i in range(1, n + 1):
        sub_file_path = path.join(file_dir, f'{name}_{i}.{suffix}')
        if path.exists(sub_file_path):
            df_ = pd.read_csv(sub_file_path)
            df = df.append(df_, ignore_index=True)
        else:
            raise Exception(f'no file: {sub_file_path}')

    if need_save:
        df.to_csv(file_path)

    return df


class Preprocess(BaseProcess):
    """
    生成预处理好的数据(计算指标值)
    """

    def __init__(self,
                 q: multiprocessing.Queue,  # queue
                 ):
        """Constructor"""

        BaseProcess.__init__(self, q, f'Preprocess',)
        self.df_source: pd.DataFrame = None


    def run(self) -> None:
        """运行"""
        self.log(f'进程启动')

        # 保存的数据文件地址
        ok_data_filepath = path.join(bt_dir, f'{ds_name}.csv')

        # 读取原始数据
        df = self.__get_source_data()
        if df is None or len(df) == 0:
            self.log(f'__get_source_data err', logging.ERROR)
            raise Exception(f'__get_source_data err')
        else:
            self.df_source = df  # todo del



def do_it(sd_name: str,  # 数据源名
          source_bar_interval: Sec,
          ):
    """生成需要的各周期数据 及 指标值"""
    need_t = [Sec.MIN1, Sec.MIN5, Sec.HOUR1][2:3]
    for t in need_t:
        f_name = f'{sd_name}_to_min{int(t.value / 60)}.csv'
        filepath = path.join(BACK_TESTING_SOURCE_DATA_DIR, f_name)
        if not path.exists(filepath):
            # 不存在，则重新生成数据
            f_source_name = f'{sd_name}.txt'
            bg_queue: Queue = Queue()
            bg = BarGenerator2(bg_queue,
                               f_name=f_name,
                               f_source_name=f_source_name,
                               source_bar_interval=source_bar_interval.value,
                               new_bar_interval=t.value,

                               new_bar_night_open=datetime.time(17),)
            bg.start()
            print(f'等待bar处理好...')
            ret = bg_queue.get()
            print(f'bar处理好了...', ret)
        else:
            # 之前有保存，就直接用
            print(f'{f_name}原先有，直接读取使用')
            df = pd.read_csv(filepath)[:]


if __name__ == '__main__':
    # df = merge_csv_files('strategy_tri_ma_min1_0404_YMH21',
    #                 file_name='min1.csv', n=12, need_save=True)
    # print(df)

    source_data: str = 'YM-CBOT-201909-202008-min1-test'  # 数据源名字
    do_it(sd_name=source_data,
          source_bar_interval=Sec.MIN1, )
