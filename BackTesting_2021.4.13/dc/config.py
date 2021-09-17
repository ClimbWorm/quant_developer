#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 数据源 配置文件
#
# Author: Tayii
# Data : 2021/1/25
# ----------------------------------------------------
import datetime
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Dict
from os import path

from constant import Sec

# SC
SC_DATA_DIR = 'E:\workspace\ScData'
SC_BAR_DATA_Columns: List[str] = ['Date', 'Time', 'Open', 'High', 'Low',
                                  'Last', 'Volume', 'NumberOfTrades',
                                  'BidVolume', 'AskVolume']

# 获取当前文件目录
DC_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DataSourceConfig:
    symbol: str
    filepath: str  # 路径
    source: str  # 数据来源
    columns: List[str]  # 源数据对应的列名
    interval: int  # 时间间隔 秒


def get_sc_min1_source_config(symbol: str,
                              filename: str,
                              ) -> DataSourceConfig:
    """获取SC的1分钟数据源配置"""
    return DataSourceConfig(
        symbol=symbol,
        filepath=path.join(SC_DATA_DIR, f'{filename}.txt'),
        source='sc',  # 数据来源
        columns=SC_BAR_DATA_Columns,  # 源数据对应的列名
        interval=Sec.MIN1.value,  # 时间间隔 秒
    )


# 数据源 配置文件
YMH21_1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='YMH21',
    filename='YMH21-CBOT.scid_BarData',
)
YM_2019year1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='YM',
    filename='YMH21-CBOT201909-202008.scid_BarData',
)
ESH21_1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='ESH21',
    filename='ESH21-CME.scid_BarData',
)
B6H21_1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='6BH21',
    filename='6BH21-CME.scid_BarData',
)
E6H21_1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='6EH21',
    filename='6EH21-CME.scid_BarData',
)

M18YM_1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='YM',
    filename='YM-CBOT_18M.scid_BarData',
)
M18ES_1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='ES',
    filename='ES-CME_18M.scid_BarData',
)
M186B_1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='6B',
    filename='6B-CME_18M.scid_BarData',
)
M186E_1_SC = get_sc_min1_source_config(  # 1分钟数据源
    symbol='6E',
    filename='6EH21-CME.scid_BarData',
)