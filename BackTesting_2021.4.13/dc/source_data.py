#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 获取处理好的原始数据
# 统一格式

# Author: Tayii
# Data : 2021/1/12
# ----------------------------------------------------
from datetime import datetime
from typing import Tuple, Optional

from dc.config import DataSourceConfig
from dc.sc import get_bar_data_from_txt
import pandas as pd

from utility import catch_except


def get_source_data_from_config(ds_config: DataSourceConfig,  # 数据源 配置名
                                size: float = 1e6,
                                ) -> Optional[pd.DataFrame]:
    """
    根据配置 获取要回测的原始数据 OHLC等
    """
    try:
        if ds_config.source == 'sc':
            df: pd.DataFrame = get_bar_data_from_txt(ds_config, size=size)
        else:
            raise NotImplementedError()
    except Exception as e:
        print(f'get_source_data {ds_config} {e}')
        return
    else:
        # TODO 格式等判断
        return df


@catch_except()
def get_back_testing_result_data(filepath: str,  # 回测结果文件地址
                                 ) -> Optional[pd.DataFrame]:
    """
    读取回测结果文件 并处理数据格式
    """
    df = pd.read_csv(filepath).drop(columns=['Unnamed: 0'])
    df['open_datetime'] = df['open_datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['close_datetime'] = df['close_datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return df
