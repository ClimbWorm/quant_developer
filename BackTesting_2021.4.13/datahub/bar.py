#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 各类bar 及 相关图形
#
# Author: Tayii
# Contact: tayiic@gmail.com
# Data : 2021/04/13
# ----------------------------------------------------
from dataclasses import dataclass, field

from constant import Direction, Sec


@dataclass
class Bar:
    """
    bar
    """
    index: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    period: Sec  # 周期 秒
    range: float = field(init=False)
    direction: Direction = field(init=False)  # 方向

    def __post_init__(self):
        self.range = self.high - self.low
        direction = (Direction.LONG if self.close >= self.open
                     else Direction.SHORT)


@dataclass
class PinBar:
    """
    判断 pin bar
    """
    ratio_threshold: float = 3  # 占比判断门限 公用
    amp_p_threshold: float = 1  # 幅度%判断门限 公用

    def is_pin_bar(self,
                   curr_bar: Bar,  # 当前bar
                   prior_bar: Bar = None,  # 前bar
                   use_bars: int = 1,  # 使用几个bar来判断
                   ratio_threshold: float = None,  # 占比判断门限
                   amp_p_threshold: float = None,  # 幅度%判断门限
                   ):
        """当前bar（或组合）是不是pin bar"""
        rt = ratio_threshold or self.ratio_threshold
        apt = amp_p_threshold or self.amp_p_threshold
        if use_bars == 1:
            high = curr_bar.high
            low = curr_bar.low
            open = curr_bar.open
        elif use_bars == 2:
            if prior_bar is None:
                return False
            high = max(curr_bar.high, prior_bar.high)
            low = min(curr_bar.low, prior_bar.low)
            open = prior_bar.open
        else:
            raise Exception(f'use_bars > 1')

        range_ = high - low
        if range_ / low < apt / 100:
            return False

        # 判断向上的pin bar
        dl = min(open, curr_bar.close)
        

        if ((curr_bar.close >= open and
             range_ / (high - curr_bar.close) > rt)
                or (curr_bar.close < open and
                    range_ / (curr_bar.close - low) > rt)):
            return True
        else:
            return False
