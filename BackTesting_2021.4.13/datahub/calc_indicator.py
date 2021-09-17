#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 单纯的 计算指标

# Author: Tayii
# Data : 2021/02/20
# ----------------------------------------------------
import datetime
from typing import Any, List, Union
import numpy

import pandas as pd

from datahub.zigzag import ZZPT
from dc.config import DataSourceConfig
from dc.sc import get_bar_data_from_txt
from datahub.indicator import IndicatorCalc as Ind


def calc_indicators(df_: pd.DataFrame,  # 原始数据
                    indicators: dict,  # 需要计算的指标
                    dynamic_zigzag: bool = False,  # zigzag需要动态参数
                    dynamic_zigzag_p: float = 0.3,  # zigzag动态比例 对应day range
                    ) -> pd.DataFrame:
    """计算指标值"""
    df = df_.copy()
    day_ranges: List[float] = []

    for index in range(len(df)):
        # 计算各指标 值放入df，保证指标值在新bar当前状态下正常
        for indicator, handle in indicators.items():

            if dynamic_zigzag and indicator == 'zigzag':
                # 在计算zigzag前更新一下day range
                if index == 0 or df.iloc[index - 1]['m_Date'] != df.iloc[index]['m_Date']:
                    day_ranges.append(df.iloc[index]['day_range'])
                else:
                    day_ranges[-1] = df.iloc[index]['day_range']  # 更新
                # 使用动态day range
                if len(day_ranges) >= 6:
                    day_range_avg: float = numpy.mean(day_ranges[-6:-1])  # 前5天的均值
                    dynamic_zigzag_day_range = day_range_avg * dynamic_zigzag_p
                    handle = lambda _df: Ind.zigzag(_df, dynamic_zigzag_day_range, mode='amount')

            r: Any = handle(df[:index + 1])

            # 一系列指标 只更新current bar
            if isinstance(r, pd.Series):
                for ind, v in r.items():
                    df.loc[index, ind] = v
            # 一系列指标，且对多bar(index)操作 其中一个是self.__new_bars_index
            elif isinstance(r, list):
                for s in r:
                    if isinstance(s, pd.Series):
                        change_bar_index = s['change_bar_index']
                        s_ = s.copy().drop('change_bar_index')
                        for ind in s_.index:
                            df.loc[change_bar_index, ind] = s_[ind]
            # 一系列指标
            elif isinstance(r, dict):
                for ind in r.keys():
                    df.loc[index, ind] = r[ind]
            # 单一指标
            else:
                df.loc[index, indicator] = r

        if index % 100 == 0:
            print(f'已处理完{index}/{len(df)} ...')

    print(day_ranges)
    return df


def get_source_data(hours_offset: float = 0,  # 偏移开盘时间
                    size: float = 1e6,
                    ) -> pd.DataFrame:
    """ 获取 计算处理的原始数据 """
    ds_config = DataSourceConfig(
        symbol='ES',
        filepath='C://Users//sc//Documents//ESH21-CME_GraphData.txt',
        source='sc',
        columns=['Date', 'Time', 'Open', 'High', 'Low', 'Last', 'Volume',
                 'ofTrades', 'OHLC_Avg', 'HLC_Avg', 'HL_Avg', 'Bid_Volume', 'Ask_Volume',
                 'ZigZag', 'TextLabels', 'ZigZagLineLength', 'ExtensionLines',
                 'BarNoInTrend', 'AskVol', 'BidVol', 'Trades', 'cpAskVol', 'cpBidVol',
                 'Duration', 'AskVol/T', 'BidVol/T', 'DeltaP', 'UpDownDelta',
                 'ATR', 'ATRSameTrend', 'ATR/T', 'BigOrder', 'VOI', 'WeightedVOI', 'PressRatio', 'DeltaSign'],
        interval=0,
    )

    df: pd.Dataframe = get_bar_data_from_txt(ds_config, size=size)

    # 加一列，偏移开盘时间后 属于哪天 ---------
    t_offset = hours_offset * 60 * 60
    df['m_Date'] = df['timestamp'].apply(
        lambda t: datetime.datetime.fromtimestamp(t - t_offset).date())
    return df[['Open', 'High', 'Low', 'Last', 'm_Date', 'datetime']]


def calc_dynamic_zigzag(
        init_reversal: float = 0.2,  # 初始几天的
        init_mode: str = 'ratio',  # 初始几天的
        dynamic_zigzag_p: float = 0.3,  # zigzag动态比例 对应day range
):
    """动态计算zigzag"""
    data = get_source_data(size=1e7)

    indicators = {  # 需要处理的数据指标
        'day_extremum': lambda df: Ind.day_extremum(df),
        'zigzag': lambda df: Ind.zigzag(df, init_reversal, mode=init_mode),  # amount
    }

    ret = calc_indicators(data[:],
                          indicators,
                          dynamic_zigzag=True,
                          dynamic_zigzag_p=dynamic_zigzag_p)
    # 获取zigzag赋值点
    selected_ret = ret[(ret['zzp_type'] == ZZPT.LOW) | (ret['zzp_type'] == ZZPT.HIGH)]
    print(selected_ret, len(selected_ret))
    ret.to_csv(f'zigzag_20210220.csv')


if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    calc_dynamic_zigzag(dynamic_zigzag_p=0.35)
