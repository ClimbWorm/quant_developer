#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 指标库
# Author: Tayii
# Data : 2021/1/28
# ----------------------------------------------------
import math

import pandas as pd
import talib
from typing import Union, Tuple, Dict

from datahub.zigzag import ZigZag


class IndicatorCalc(object):
    """
    计算指标 (按首字母排序)

    # 公用参数说明
        data: pd.DataFrame, # 原始数据
        whole: bool, # false==生成最后单个bar数据  true==全部bar对应的指标值
    """

    @classmethod
    def adx(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col_high: str = 'High',  # 计算的列名
            col_low: str = 'Low',  # 计算的列名
            col_close: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 ADX"""
        result = talib.ADX(data[col_high], data[col_low], data[col_close], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def adxr(cls, data: pd.DataFrame,  # 原始数据
             n: int,  # 周期
             col_high: str = 'High',  # 计算的列名
             col_low: str = 'Low',  # 计算的列名
             col_close: str = 'Last',  # 计算的列名
             whole: bool = False,
             ) -> Union[float, pd.Series]:
        """计算 ADXR"""
        result = talib.ADXR(data[col_high], data[col_low], data[col_close], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def atr(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col_high: str = 'High',  # 计算的列名
            col_low: str = 'Low',  # 计算的列名
            col_close: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 ATR"""
        result = talib.ATR(data[col_high], data[col_low], data[col_close], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def boll(cls, data: pd.DataFrame,  # 原始数据
             n: int = 20,  # 周期
             dev: float = None,  # 偏移  ==None时候输出标准差 不输出上下line
             ma_type: str = 'sma',  # ma类型 EMA SMA
             col: str = 'Last',  # 计算的列名
             whole: bool = False,
             ) -> Union[Dict[str, pd.Series], Dict[str, float]]:
        """
        Bollinger Channel.
        """
        if ma_type == 'sma':
            mid = cls.sma(data, n, col=col, whole=whole)
        elif ma_type == 'ema':
            mid = cls.ema(data, n, col=col, whole=whole)
        else:
            raise Exception(f'不支持的ma_type {ma_type}')

        std = cls.std(data, n, col=col, whole=whole)

        if dev:
            up = mid + std * dev
            down = mid - std * dev
            return {f'boll_up_{ma_type}_{n}': up, f'boll_down_{ma_type}_{n}': down,
                    f'boll_mid_{ma_type}_{n}': mid, }
        else:
            return {f'boll_mid_{ma_type}_{n}': mid, f'boll_std_{ma_type}_{n}': std}

    @classmethod
    def cci(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 CCI"""
        result = talib.CCI(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def cmo(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 CMO"""
        result = talib.CMO(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def day_extremum(cls, data: pd.DataFrame) -> dict:
        """计算日内高低点 振幅"""
        n = data.index[-1]
        # 不同日期
        if n == 0 or data.loc[n, 'm_Date'] != data.loc[n - 1, 'm_Date']:
            day_high, day_low = data.loc[n, 'High'], data.loc[n, 'Low']
        else:
            day_high = max(data.loc[n, 'High'], data.loc[n - 1, 'day_high'])
            day_low = min(data.loc[n, 'Low'], data.loc[n - 1, 'day_low'])
        return {'day_high': day_high, 'day_low': day_low,
                'day_range': day_high - day_low, }

    @classmethod
    def days_avg_range(cls, data: pd.DataFrame,
                       p: int = 5,  # 几个周期的平均
                       ) -> pd.Series:
        """计算N日 平均振幅"""
        grouped = data.groupby('m_Date')['day_range'].max()
        df = pd.DataFrame(grouped)

        avg_result = []
        for i in range(len(df)):
            avg_result.append(df['day_range'][max(0, i - p + 1): i + 1].mean())
        df['avg_range'] = avg_result

        avg_range_ratio = []
        for i in range(len(df)):
            if i < p:  # 前几天不是p天平均值 可能偏小
                avg_range_ratio.append(df.iloc[i]['avg_range'] / df['avg_range'][: i + 1].min())
            else:
                avg_range_ratio.append(df.iloc[i]['avg_range'] / df['avg_range'][p: i + 1].min())

        df['avg_range_ratio'] = avg_range_ratio

        return df['avg_range_ratio']

    @classmethod
    def dx(cls, data: pd.DataFrame,  # 原始数据
           n: int,  # 周期
           col_high: str = 'High',  # 计算的列名
           col_low: str = 'Low',  # 计算的列名
           col_close: str = 'Last',  # 计算的列名
           whole: bool = False,
           ) -> Union[float, pd.Series]:
        """计算 DX"""
        result = talib.DX(data[col_high], data[col_low], data[col_close], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def ema(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 EMA"""
        result = talib.EMA(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def heikin_ashi(cls, df: pd.DataFrame,  # 原始数据
                    ) -> Union[float, pd.Series]:
        """ 计算 Heikin-Ashi 蜡烛图 """
        curr = df.index[-1]
        if len(df) < 2:
            close, close_hlc, close_hlcc, open, high, low = 0, 0, 0, 0, 0, 0
        else:
            prior = curr - 1
            close = (df.loc[curr, 'High'] + df.loc[curr, 'Low'] + df.loc[curr, 'Open']
                     + df.loc[curr, 'Last']) / 4
            close_hlc = (df.loc[curr, 'High'] + df.loc[curr, 'Low']
                         + df.loc[curr, 'Last']) / 3
            close_hlcc = (df.loc[curr, 'High'] + df.loc[curr, 'Low']
                          + 2 * df.loc[curr, 'Last']) / 4
            open = (df.loc[prior, 'Open'] + df.loc[curr, 'Open']) / 2
            high = max(df.loc[curr, 'High'], close, open)
            low = min(df.loc[curr, 'Low'], close, open)

        return {f'ha_close': close,
                f'ha_close_hlc': close_hlc,
                f'ha_close_hlcc': close_hlcc,
                f'ha_open': open,
                f'ha_high': high,
                f'ha_low': low,
                }

    @classmethod
    def kama(cls, data: pd.DataFrame,  # 原始数据
             n: int,  # 周期
             col: str = 'Last',  # 计算的列名
             whole: bool = False,
             ) -> Union[float, pd.Series]:
        """计算 KAMA"""
        result = talib.KAMA(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def minus_di(cls, data: pd.DataFrame,  # 原始数据
                 n: int,  # 周期
                 col_high: str = 'High',  # 计算的列名
                 col_low: str = 'Low',  # 计算的列名
                 col_close: str = 'Last',  # 计算的列名
                 whole: bool = False,
                 ) -> Union[float, pd.Series]:
        """计算 MINUS_DI"""
        result = talib.MINUS_DI(data[col_high], data[col_low], data[col_close], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def minus_dm(cls, data: pd.DataFrame,  # 原始数据
                 n: int,  # 周期
                 col_high: str = 'High',  # 计算的列名
                 col_low: str = 'Low',  # 计算的列名
                 whole: bool = False,
                 ) -> Union[float, pd.Series]:
        """计算 MINUS_DM"""
        result = talib.MINUS_DM(data[col_high], data[col_low], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def mom(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 MOM"""
        result = talib.MOM(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def natr(cls, data: pd.DataFrame,  # 原始数据
             n: int,  # 周期
             col: str = 'Last',  # 计算的列名
             whole: bool = False,
             ) -> Union[float, pd.Series]:
        """计算 NATR"""
        result = talib.NATR(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def obv(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 OBV"""
        result = talib.OBV(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def plus_di(cls, data: pd.DataFrame,  # 原始数据
                n: int,  # 周期
                col_high: str = 'High',  # 计算的列名
                col_low: str = 'Low',  # 计算的列名
                col_close: str = 'Last',  # 计算的列名
                whole: bool = False,
                ) -> Union[float, pd.Series]:
        """计算 PLUS_DI"""
        result = talib.PLUS_DI(data[col_high], data[col_low], data[col_close], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def plus_dm(cls, data: pd.DataFrame,  # 原始数据
                n: int,  # 周期
                col_high: str = 'High',  # 计算的列名
                col_low: str = 'Low',  # 计算的列名
                whole: bool = False,
                ) -> Union[float, pd.Series]:
        """计算 PLUS_DM"""
        result = talib.PLUS_DM(data[col_high], data[col_low], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def roc(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 ROC"""
        result = talib.ROC(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def rocr(cls, data: pd.DataFrame,  # 原始数据
             n: int,  # 周期
             col: str = 'Last',  # 计算的列名
             whole: bool = False,
             ) -> Union[float, pd.Series]:
        """计算 ROCR"""
        result = talib.ROCR(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def rocr100(cls, data: pd.DataFrame,  # 原始数据
                n: int,  # 周期
                col: str = 'Last',  # 计算的列名
                whole: bool = False,
                ) -> Union[float, pd.Series]:
        """计算 ROCR100"""
        result = talib.ROCR100(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def rocp(cls, data: pd.DataFrame,  # 原始数据
             n: int,  # 周期
             col: str = 'Last',  # 计算的列名
             whole: bool = False,
             ) -> Union[float, pd.Series]:
        """计算 ROCP"""
        result = talib.ROCP(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def rsi(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 RSI"""
        result = talib.RSI(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def sma(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 EMA"""
        result = talib.SMA(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def std(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 STD"""
        result = talib.STDDEV(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def trange(cls, data: pd.DataFrame,  # 原始数据
               n: int,  # 周期
               col_high: str = 'High',  # 计算的列名
               col_low: str = 'Low',  # 计算的列名
               col_close: str = 'Last',  # 计算的列名
               whole: bool = False,
               ) -> Union[float, pd.Series]:
        """计算 TRANGE"""
        result = talib.TRANGE(data[col_high], data[col_low], data[col_close], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def trix(cls, data: pd.DataFrame,  # 原始数据
             n: int,  # 周期
             col: str = 'Last',  # 计算的列名
             whole: bool = False,
             ) -> Union[float, pd.Series]:
        """计算 TRIX"""
        result = talib.TRIX(data[col], n)
        return result if whole else result.iloc[-1]

    # @classmethod
    # def ultosc(cls, data: pd.DataFrame,  # 原始数据 # TODO 修改
    #             n: int,  # 周期
    #             col_high: str = 'High',  # 计算的列名
    #             col_low: str = 'Low',  # 计算的列名
    #             col_close: str = 'Last',  # 计算的列名
    #             whole: bool = False,
    #             ) -> Union[float, pd.Series]:
    #     """计算 PLUS_DI"""
    #     result = talib.PLUS_DI(data[col_high], data[col_low], data[col_close], n)
    #     return result if whole else result.iloc[-1]

    @classmethod
    def willr(cls, data: pd.DataFrame,  # 原始数据
              n: int,  # 周期
              col_high: str = 'High',  # 计算的列名
              col_low: str = 'Low',  # 计算的列名
              col_close: str = 'Last',  # 计算的列名
              whole: bool = False,
              ) -> Union[float, pd.Series]:
        """计算 WILLR"""
        result = talib.WILLR(data[col_high], data[col_low], data[col_close], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def wma(cls, data: pd.DataFrame,  # 原始数据
            n: int,  # 周期
            col: str = 'Last',  # 计算的列名
            whole: bool = False,
            ) -> Union[float, pd.Series]:
        """计算 WMA"""
        result = talib.WMA(data[col], n)
        return result if whole else result.iloc[-1]

    @classmethod
    def zigzag(cls, data: pd.DataFrame,
               reversal: Union[int, float],
               mode: str = 'ratio',
               ) -> list:
        """计算 zigzag"""
        return ZigZag.calc(data, reversal, mode=mode)
