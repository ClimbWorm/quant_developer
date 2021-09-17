#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# ZigZag
#
# Author: Tayii
# Contact: tayiic@gmail.com
# Data : 2020/12/21
# ----------------------------------------------------
import pandas as pd
import numpy as np
from typing import Union
from enum import Enum
from dataclasses import dataclass


class ZZPT(Enum):
    """
    ZigZag赋值点类型
    """
    HIGH = 'High ZZP'
    ONCE_HIGH = 'Once High ZZP'  # 曾经是
    LOW = 'Low ZZP'
    ONCE_LOW = 'Once Low ZZP'  # 曾经是
    NOT_ZZP = "Not ZZP"


@dataclass()
class ZZP_BAR(object):
    """
    带有ZigZag赋值点 结构的bar
    """
    type_: ZZPT = ZZPT.NOT_ZZP  # 高点还是低点
    bar_index: int = -1  # bar index
    length: float = 0.0  # 到上个赋值点的垂直距离，有正负
    extremum: float = 0.0  # 赋值点的极值 即高点的high 低点的low值
    threshold: float = 0.0  # 反转的阈值 (动态调整时用 仅在amount模式下有效）

    def update(self, type_: ZZPT, bar_index: int, length: float,
               extremum: float, threshold: float = 0):
        self.type_ = type_
        self.bar_index = bar_index
        self.length = length
        self.extremum = extremum
        self.threshold = threshold  # 仅在amount模式下有效


class ZigZag(object):
    """
    ZigZag 计算
    """

    @classmethod
    def read_bar_from_df(cls, n: int, df: pd.DataFrame) -> ZZP_BAR:
        """从dataframe中读取bar(index=n)的信息 并转换为ZZP—bar结构体"""
        zzp = ZZP_BAR()
        s = df.loc[n]
        zzp.update(s['zzp_type'], n, s['zzp_length'], s['zzp_extremum'], s['threshold'])
        return zzp

    @classmethod
    def zzp_to_series(cls, type_: ZZPT, bar_index: int, length: float,
                      extremum: float, threshold: float = 0):
        """ 转换为series """
        return pd.Series({'zzp_type': type_,
                          'change_bar_index': bar_index,
                          'zzp_length': length,
                          'zzp_extremum': extremum,
                          'threshold': threshold,
                          })

    @classmethod
    def calc(cls, df: Union[pd.DataFrame, np.array],
             reversal: Union[int, float],
             mode: str = 'ratio',
             threshold_up_limit: float = None  # 反转的最大阈值 0.01=1%*open价格
             ) -> list:
        """
        Args:
            df: 输入的数据 i.e.从原始数据切片到当前要处理的index
            reversal:
                “ratio”模式下：反转的比例 %（可设0.1-1，为百分比；小于0.1默认会*100，即0.001=0.1%）
                ”amount“模式下：反转的实际数值
            mode: 阈值模式； ratio=按比例， amount=按数值
        Returns:
            [pd.Seires,...]  一个合作多个，都用list
        """
        if df is None or len(df) == 0:
            return []

        n = df.index[-1]  # 当前计算的bar index, 即最后一行

        if mode == 'ratio':
            assert 0 < reversal < 100, '需要满足 0 < reversal_percent < 100'
            reversal = reversal if reversal < 0.1 else reversal / 100.0
            amount = None
        else:
            amount = (min(reversal, threshold_up_limit * df.loc[n, 'Open'])
                      if threshold_up_limit else reversal)  # 反转量

        # 第一行（第一个bar）
        if n == df.index[0]:
            return [cls.zzp_to_series(ZZPT.NOT_ZZP, n, 0, 0, amount if amount else 0)]

        # 第二个bar以后 ------------------------------------

        prior_bar: ZZP_BAR = cls.read_bar_from_df(n - 1, df)

        def _find_nearest_zzp():
            """找出 当前bar之前 最近的极值点"""
            for i in range(len(df) - 2, -1, -1):  # 不包括当前bar
                if df.iloc[i]['zzp_extremum'] > 0:
                    return i
            return -1  # 没有找到

        # 当前bar之前 最近的极值点
        prior_zzp_index = _find_nearest_zzp()

        # 刚开始那一段  # 即前面没有赋值点 -------------
        if prior_zzp_index == -1:
            high, low = df['High'].max(), df['Low'].min()

            # 对当前点 高低点都检查
            for t in ['High', 'Low']:
                threshold = amount if amount else reversal * (low if t == 'High' else high)
                curr_extremum = df.loc[n, ('High' if t == 'High' else 'Low')]
                _length = curr_extremum - low if t == 'High' else high - curr_extremum
                if _length > threshold:
                    # TODO 可以考虑再反推一个赋值点，但也没多大必要
                    return [cls.zzp_to_series(ZZPT.HIGH if t == 'High' else ZZPT.LOW,
                                              n, _length, curr_extremum, threshold, )]

            # 高低点都不满足
            return [cls.zzp_to_series(ZZPT.NOT_ZZP, n, 0, 0, threshold), ]

        # 前面已经有赋值点 ----------------------
        else:
            prior_zzp = cls.read_bar_from_df(prior_zzp_index, df)  # 前赋值点
            prior_zzp_is_high = prior_zzp.type_ == ZZPT.HIGH  # 前赋值点是否是高点

            # 先判断有没有漂移，有就不考虑反转了
            diff = df.loc[n, 'High'] - prior_zzp.extremum if prior_zzp_is_high \
                else prior_zzp.extremum - df.loc[n, 'Low']
            if diff > 0:
                return [  # 当前点设置，
                    cls.zzp_to_series(ZZPT.HIGH if prior_zzp_is_high else ZZPT.LOW,
                                      n,  # 修改的位置 index
                                      diff + prior_zzp.length,
                                      df.loc[n, ('High' if prior_zzp_is_high else 'Low')],
                                      df.loc[prior_zzp.bar_index, 'threshold'], ),
                    # 前赋值点修改为ONCE
                    cls.zzp_to_series(ZZPT.ONCE_HIGH if prior_zzp_is_high else ZZPT.ONCE_LOW,
                                      bar_index=prior_zzp.bar_index,  # 同时修改 前赋值点
                                      length=0,
                                      extremum=(df.loc[prior_zzp.bar_index, 'High']
                                                if prior_zzp_is_high
                                                else df.loc[prior_zzp.bar_index, 'Low']),
                                      threshold=df.loc[prior_zzp.bar_index, 'threshold'], ),
                ]

            # 判断有没有反转
            threshold = amount if amount else reversal * prior_zzp.extremum
            curr_extremum = df.loc[n, ('Low' if prior_zzp_is_high else 'High')]
            _length = prior_zzp.extremum - curr_extremum if prior_zzp_is_high else curr_extremum - prior_zzp.extremum
            if _length > threshold:
                return [cls.zzp_to_series(ZZPT.LOW if prior_zzp_is_high else ZZPT.HIGH,
                                          n,
                                          _length,
                                          curr_extremum,
                                          threshold,
                                          ), ]
            else:  # 不满足
                return [cls.zzp_to_series(ZZPT.NOT_ZZP, n, 0, 0, 0), ]


# class ZigZagOld(threading.Thread):
#     def __init__(self, init_data: Union[pd.DataFrame, np.array], reversal: Union[int, float],
#                  mode: str = 'ratio', q: Queue = None) -> None:
#         """
#         Args:
#             init_data: 输入的数据
#             mode: 阈值模式； ratio=按比例， amount=按数值
#             reversal:
#                 ratio模式下：反转的比例 %（可设0.1-1，为百分比；小于0.1默认会*100，即0.001=0.1%）
#                 amount模式下：反转的实际数值
#             q: 通信用的queue
#         """
#         threading.Thread.__init__(self)
#         self.name = 'ZigZag'
#
#         if isinstance(init_data, np.ndarray):
#             try:
#                 self._df = pd.DataFrame(init_data, columns=['GMT', 'Open', 'High', 'Low', 'Close', 'Volume'])
#             except Exception as e:
#                 raise Exception(f'Input data err')
#         else:
#             assert isinstance(init_data, pd.DataFrame)
#             self._df = init_data
#
#         assert mode in ['ratio', 'amount']
#         self.mode = mode
#
#         if mode == 'ratio':
#             assert 0 < reversal < 100, '需要满足 0 < reversal_percent < 100'
#             self._rp = reversal if reversal < 0.1 else reversal / 100.0
#         else:  # mode == 'amount'
#             self.amount = reversal
#
#         self._q = q
#         self._fixed_zzps = pd.DataFrame(columns=['type_', 'bar_num', 'length', 'extremum'])
#         self._moving_zzp = ZZP_BAR()  # 最后（最新）一个赋值点，可能漂移的
#
#         if len(init_data) > 1:
#             self.calculate_with_all_data()
#
#     def run(self) -> None:
#         """处理持续更新的数据"""
#         pass
#
#     def calculate_with_all_data(self) -> None:
#         """根据所有数据 从头开始计算一遍"""
#         df = self._df.copy()
#
#         # 初始化阶段，找第一个赋值点
#         high, low = df.iloc[0]['High'], df.iloc[0]['Low']
#         for index, row in df.iterrows():
#             _length = high * self._rp if self.mode == 'ratio' else self.amount
#             if row['Low'] <= high - _length:
#                 self._move_zzp(ZZPT.LOW, index, row['Low'] - high, row['Low'])
#                 break
#
#             _length = low * self._rp if self.mode == 'ratio' else self.amount
#             if row['High'] >= low + _length:
#                 self._move_zzp(ZZPT.HIGH, index, row['High'] - low, row['High'])
#                 break
#
#         # 前面已经有赋值点
#         m_zzp = self._moving_zzp
#         _length = m_zzp.extremum * self._rp if self.mode == 'ratio' else self.amount
#         for index, row in df[self._moving_zzp.bar_index + 1:].iterrows():
#             if m_zzp.type_ == ZZPT.HIGH:
#                 # 先判断有没有漂移，有就不考虑反转了
#                 if row['High'] > m_zzp.extremum:
#                     _new_length = row['High'] - m_zzp.extremum + m_zzp.length
#                     self._move_zzp(ZZPT.HIGH, index, _new_length, row['High'])
#                 # 判断有没有反转
#                 elif row['Low'] <= m_zzp.extremum - _length:
#                     self._save_fixed_zzp(m_zzp)
#                     self._move_zzp(ZZPT.LOW, index, row['Low'] - m_zzp.extremum, row['Low'])
#             elif m_zzp.type_ == ZZPT.LOW:
#                 # 先判断有没有漂移
#                 if row['Low'] < m_zzp.extremum:
#                     _new_length = row['Low'] - m_zzp.extremum + m_zzp.length
#                     self._move_zzp(ZZPT.LOW, index, _new_length, row['Low'])
#                 # 判断有没有反转
#                 elif row['High'] >= m_zzp.extremum + _length:
#                     self._save_fixed_zzp(m_zzp)
#                     self._move_zzp(ZZPT.HIGH, index, row['High'] - m_zzp.extremum, row['High'])
#
#     def _move_zzp(self, type_, index, length, extremum):
#         """更新（移动） zigzag赋值点"""
#         self._moving_zzp.update(type_, index, length, extremum)
#         # 保存漂移点
#         m_type = ZZPT.ONCE_HIGH if type_ == ZZPT.HIGH else ZZPT.ONCE_LOW
#         self._save_fixed_zzp(ZZP_BAR(m_type, index, 0.0, extremum))
#
#     def _save_fixed_zzp(self, zzp: ZZP_BAR):
#         """保存固定了的zigzag赋值点"""
#         self._fixed_zzps = self._fixed_zzps.append(pd.Series(zzp.__dict__), ignore_index=True)
#
#     def fixed_zzps(self, add_once=False):
#         """
#         获取所有固定了的zigzag赋值点
#         Args:
#             add_once: 是否添加 曾经的赋值点（临时赋值点）
#         Returns:
#             Dataframe
#         """
#         df = self._fixed_zzps
#         if not add_once:
#             df = df[df['type_'].isin([ZZPT.HIGH, ZZPT.LOW])]
#         return df.reset_index(drop=True)
#
#     def all_zzp(self, add_once=False):
#         """获取所有zigzag赋值点"""
#         df = self.fixed_zzps(add_once=add_once)
#         return df.append(pd.Series(self._moving_zzp.__dict__), ignore_index=True)


if __name__ == '__main__':
    pass
