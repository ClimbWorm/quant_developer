#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 回测 template
#
# Author: Tayii
# Data : 2021/2/2
# ----------------------------------------------------
from dataclasses import dataclass, field
import datetime
from typing import List, Dict

from dc.config import DataSourceConfig


@dataclass()
class BackTestingDataSetting:
    """
    回测数据配置
    一个配置产生一个新周期的bar数据
    """
    data_source: DataSourceConfig = None  # 数据源 配置
    day_open: float = 0.0  # 每天开盘的时间 小时计 （可选）
    symbol: str = field(init=False)
    source_data_interval: int = field(init=False)  # 获取源数据的周期
    new_bar_interval: int = 0  # 新bar 周期
    new_bar_day_open: datetime.time = datetime.time(0, )  # 新bar 日盘开盘时间
    new_bar_night_open: datetime.time = None  # 新bar 夜盘开盘时间
    need_columns: List = None  # 策略需要的数据（列名）
    indicators: Dict = None  # 计算的指标

    def __post_init__(self):
        self.symbol = self.data_source.symbol
        self.source_data_interval = self.data_source.interval

    def __str__(self):
        return f'DataSetting: {self.symbol}_{self.new_bar_interval}'

    __repr__ = __str__


if __name__ == '__main__':
    pass
