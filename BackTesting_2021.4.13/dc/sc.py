#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 
#
# Author: Tayii
# Contact: tayiic@gmail.com
# Data : 2020/12/10
# ----------------------------------------------------
from typing import List, Union

import pandas as pd
from datetime import datetime

from dc.config import DataSourceConfig
from utility import str2timestamp, catch_except


def get_txt_data(filepath: str,  # 文件地址
                 size: float = 1e6,  # 最大上限（行数）
                 header: Union[int, str, List[int]] = None,  # 列名 default ‘infer’
                 ) -> pd.DataFrame:
    """ 获取txt数据 """
    df = pd.DataFrame()
    _df = pd.read_table(filepath, sep=', ', header=header, error_bad_lines=False,
                        chunksize=1e5)
    for chunk in _df:
        df = df.append(chunk)
        if len(df) > size:
            break
    return df


# @timeit
@catch_except()
def get_bar_data_from_txt(ds_config: DataSourceConfig = None,
                          filepath: str = None,  # 文件路径 与上面config二选一
                          size: float = 1e6,
                          header: Union[int, str, List[int]] = None,  # 列名
                          columns: List[str] = None
                          ) -> pd.DataFrame:
    """ 获取SC导出txt的bar数据 """
    filepath = ds_config.filepath if ds_config else filepath
    df = get_txt_data(filepath, size=size, header=header)

    if ds_config or columns:
        # 特殊处理
        df = df[1:]  # 去掉标签行
        df.columns = ds_config.columns if ds_config else columns

    for c in df.columns[2:]:
        df[c] = pd.to_numeric(df[c], errors='raise')  # 字符串转数字

    # # 新增 时间戳
    df['Date'] = df['Date'].apply(lambda x: x.replace('/', '-'))
    df['Time'] = df['Time'].apply(lambda x: x[:8])
    df['datetime'] = df['Date'] + ' ' + df['Time']
    df['timestamp'] = df['datetime'].apply(lambda x: str2timestamp(x, ms=False))
    df['datetime'] = df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    # df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    # df['Time'] = df['Time'].apply(lambda x: datetime.time(datetime.strptime(x, '%H:%M:%S')))

    df.sort_values('timestamp', ascending=True, inplace=True)  # 升序

    return df.reset_index(drop=True, )


def check_data(df: pd.DataFrame, interval: int):
    """
    检查bar数据 是否正常 合理
    Args:
        df: 源数据
        interval: 每一bar的周期
    Returns:

    """
    # TODO
    # df['interval'] = df['timestamp'] - df['timestamp'].shift(1)
    # ret_df = df[abs(df['shift']) > 0.5]  # 取有变动的
    pass
    # return ret_df


if __name__ == '__main__':
    pass
