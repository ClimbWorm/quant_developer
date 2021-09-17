#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 各类状态的类
#
# Author: Tayii
# Contact: tayiic@gmail.com
# Data : 2020/12/21
# ----------------------------------------------------
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List

import pandas as pd

from constant import Direction


@dataclass
class BlStatus:
    """
    记录 价格与布林带相互的状态，交叉后重新更新
    """
    len_: int  # 计算周期
    dev: float  # 偏移
    n_for_support: int  # 几个bar算强支撑线
    n_for_in_bl_t: int = None  # 在布林带里持续几个bar算超限
    ref: float = None  # 参照的价格
    mid: float = None  # 中值
    std: float = None  # std
    curr_index: int = None  # 当前bar的index

    # 动态计算更新的值
    ul: float = field(init=False)  # 上轨
    dl: float = field(init=False)  # 下轨
    up_up: bool = field(init=False)  # 是否上轨之上
    down_down: bool = field(init=False)  # 是否下轨之下
    price_bl: Direction = field(init=False)  # LONG==价格在up线上，SHORT==价格在down线下
    realtime_dev: float = None  # 价格实时的布林带偏移
    bl_start_index: int = None  # 突破时的布林带对应bar
    bl_start_value: float = field(init=False)  # 突破时（第一个bar）的布林带对应值
    bl_amp: float = field(init=False)  # 距离起点幅度 %
    bl_past: int = field(init=False)  # 距离起点过了多少bar

    start_less_up_index: int = 0  # 开始小于上轨的index
    start_more_down_index: int = 0  # 开始大于下轨的index
    prior_start_less_up_index: int = 0  # 开始小于上轨的index
    prior_start_more_down_index: int = 0  # 开始大于下轨的index
    prior_less_up_t: int = 0  # 之前轮 小于上轨的持续时间 用bar数计
    prior_more_down_t: int = 0  # 之前轮 大于下轨的持续时间 用bar数计

    up_mid: bool = field(init=False)  # 是否中轨之上
    price_mid: Direction = field(init=False)  # LONG==价格在mid线上，SHORT==价格在mid线下
    mid_start_index: int = None  # 突破mid时的布林带对应bar
    mid_start_value: float = field(init=False)  # 突破mid时的布林带对应值
    mid_amp: float = field(init=False)  # 距离起点幅度 %
    mid_past: int = field(init=False)  # 距离起点过了多少bar
    mid_past_ratio: float = 0  # 距离跟门限的比例

    prior_turn_p_bl: Direction = None  # 之前轮
    p_prior_turn_p_bl: Direction = None  # 前前轮
    prior_turn_bl_past: int = None  # 之前bar状态值 走了几个bar 一般在突破时赋值
    prior_turn_mid_past = None  # 之前轮 走了几个bar
    prior_mid_past_ratio: float = 0  # 之前轮 距离跟门限的比例

    # def __post_init__(self):
    #     self.update_bl()
    #     self.update_mid()

    def update(self,
               ref: float,  # 参照的价格
               mid: float,  # 中值
               std: float,  # std
               curr_index: int,
               ):
        self.ref = ref
        self.mid = mid
        self.std = std
        self.curr_index = curr_index
        self.update_bl()
        self.update_mid()

    def update_bl(self, ) -> None:
        """更新"""
        self.ul = self.mid + self.dev * self.std  # 上轨
        self.dl = self.mid - self.dev * self.std  # 上轨
        self.realtime_dev = (self.ref - self.mid) / self.std
        # 当前周期 上下轨与close相对位置  True==close在up上 或者 在down下
        self.up_up = True if self.ul < self.ref else False
        self.down_down = True if self.dl > self.ref else False

        # 先保存之前的信息
        try:
            prior_p_bl_ = self.price_bl
        except:
            prior_p_bl_ = None

        self.price_bl = (Direction.LONG if self.up_up else
                         (Direction.SHORT if self.down_down else Direction.NONE))

        if prior_p_bl_ is None:  # 第一次
            self.bl_start_index = self.curr_index
            self.bl_start_value = (self.ul if self.mid <= self.ref
                                   else self.dl)  # 第一次用中线判断 粗略的即可
        else:  # 更新布林带状态
            # 原本布林带外
            if prior_p_bl_ != Direction.NONE:
                # 回到布林带内 or 反穿到对面
                if self.price_bl != prior_p_bl_:
                    # 保留上轮记录
                    self.p_prior_turn_p_bl = self.prior_turn_p_bl
                    self.prior_turn_p_bl = prior_p_bl_
                    self.prior_turn_bl_past = self.bl_past - 1
                    # 更新
                    self.bl_start_index = self.curr_index
                    self.bl_start_value = (
                        (self.ul if prior_p_bl_ == Direction.LONG else self.dl)  # 回到布林带内
                        if self.price_bl == Direction.NONE else
                        (self.ul if self.price_bl == Direction.LONG else self.dl))  # 反穿到对面

                    # 更新 价格没有触碰到上轨/下轨多久
                    if prior_p_bl_ == Direction.LONG:
                        self.prior_start_less_up_index = self.start_less_up_index
                        self.start_less_up_index = self.curr_index
                    else:  # 回到下轨内（上面）
                        self.prior_start_more_down_index = self.start_more_down_index
                        self.start_more_down_index = self.curr_index

            # 原本布林带内
            else:
                # 开始突破布林带
                if self.is_out_bl_line:
                    self.bl_start_index = self.curr_index
                    self.bl_start_value = (self.ul if self.price_bl == Direction.LONG else self.dl)

                    # 更新 价格在某一侧布林带内的计数（没有触碰到上轨/下轨多久）
                    if self.price_bl == Direction.LONG:
                        t = self.curr_index - self.start_less_up_index
                        # if (self.start_less_up_index - self.prior_start_less_up_index
                        #         > self.n_for_in_bl_t):  # 忽略小震荡
                        self.prior_less_up_t = t
                    else:
                        t = self.curr_index - self.start_more_down_index
                        # if (self.start_more_down_index - self.prior_start_more_down_index
                        #         > self.n_for_in_bl_t): # 忽略小震荡
                        self.prior_more_down_t = t

        # 更新 当前状态已经过了几个bar
        self.bl_past = self.curr_index - self.bl_start_index
        if self.price_bl != Direction.NONE:
            # 上轨上时增长，下轨下时下跌，则结果为正，即同步；反之为负
            diff = (self.ul - self.bl_start_value if self.price_bl == Direction.LONG
                    else self.bl_start_value - self.dl)
            self.bl_amp = diff / self.bl_start_value * 100
        else:  # 布林带内 暂时不计算
            self.bl_amp = 0

    def update_mid(self, ) -> None:
        """更新均线/mid状态 """
        self.up_mid = True if self.mid <= self.ref else False

        # 先保存之前的信息
        try:
            prior_p_mid_ = self.price_mid
        except:
            prior_p_mid_ = None

        self.price_mid = Direction.LONG if self.up_mid else Direction.SHORT
        if prior_p_mid_ is None or self.price_mid != prior_p_mid_:
            # 第一次 或者 突破
            self.prior_turn_mid_past = self.mid_past if prior_p_mid_ else 0
            self.prior_mid_past_ratio = self.prior_turn_mid_past / self.n_for_support
            self.mid_start_index = self.curr_index
            self.mid_start_value = self.mid

        # 更新 当前状态已经过了几个bar
        self.mid_past = self.curr_index - self.mid_start_index
        self.mid_past_ratio = self.mid_past / self.n_for_support
        # 均线上时增长，均线下时下跌，则结果为正，即同步；反之为负
        diff = self.mid - self.mid_start_value
        diff = diff if self.up_mid else -diff
        self.mid_amp = diff / self.mid_start_value * 100

    @property
    def now_less_up_t(self) -> int:
        """小于上轨的持续时间 用bar数计"""
        if self.start_less_up_index and self.price_bl != Direction.LONG:
            return self.curr_index - self.start_less_up_index
        else:
            return 0

    @property
    def now_more_down_t(self):
        """大于下轨的持续时间 用bar数计"""
        if self.start_more_down_index and self.price_bl != Direction.SHORT:
            return self.curr_index - self.start_more_down_index
        else:
            return 0

    def how_long_in_bl(self, direction: Direction) -> int:
        """在上轨/下轨内多久了（基于当前和之前轮）"""
        assert direction != Direction.NONE
        if direction == Direction.LONG:
            return max(self.now_less_up_t, self.prior_less_up_t)
        else:
            return max(self.now_more_down_t, self.prior_more_down_t)

    @property
    def is_out_bl_line(self):
        """在布林带外"""
        return self.price_bl is not Direction.NONE

    @property
    def is_out_small_bl_line(self, multi=0.75):
        """在小倍数的布林带外"""
        small_ul = self.mid + multi * self.dev * self.std  # 小倍数的上轨
        small_dl = self.mid - multi * self.dev * self.std  # 小倍数的上轨
        return (Direction.LONG if self.ref > small_ul else
                (Direction.SHORT if self.ref < small_dl else Direction.NONE))

    @property
    def is_break_in_bl(self,
                       ) -> Direction:  # 之前price_bl方向的相反方向
        """当前bar从布林带外穿越入布林带内"""
        if self.bl_past == 0 and self.price_bl == Direction.NONE:
            # 下轨向上突破 做多，反之空
            return Direction.opposite(self.prior_turn_p_bl)
        return Direction.NONE

    @property
    def is_break_out_bl(self,
                        ) -> Direction:  # price_bl方向
        """是否 当前bar从布林带内突破到布林带外"""
        if self.bl_past == 0:
            return self.price_bl
        return Direction.NONE

    @property
    def bl_offset(self) -> float:
        """上下轨距离"""
        return self.dev * self.std * 2

    @property
    def is_support_line(self,
                        ) -> Direction:
        """
        是否是支撑均线
        Returns:
            支撑哪个方向
            Direction.NONE: 不是强支撑线
        """
        return self.price_mid if self.mid_past_ratio >= 1 else Direction.NONE

    @property
    def is_strong_support_line(self,
                               ) -> Direction:
        """
        是否是强支撑均线
        """
        support_line = self.is_support_line
        out_small_bl = self.is_out_small_bl_line
        return support_line if support_line == out_small_bl else Direction.NONE

    @property
    def cross_support_line(self,
                           ) -> Direction:
        """
        是否是支撑均线并回穿
        Returns:
            开仓方向
            Direction.NONE: 没有回穿或者回穿的不是强支撑线
        """
        if self.prior_turn_mid_past >= self.n_for_support:
            return self.is_cross_mid
        return Direction.NONE  #

    @property
    def is_cross_mid(self,
                     ) -> Direction:  # 开仓方向
        """是否正穿越mid"""
        return self.price_mid if self.mid_past == 0 else Direction.NONE


@dataclass
class MultiBlStatus:
    """
    处理 多条布林带相互的状态
    """
    min1_ma_type: str  #
    c_price_type: str  # 判断是用的实时价的类型
    all_lines: list
    support_lines: List[int] = None  # 支撑线选用的length
    bs: Dict[int, BlStatus] = None  # 各布林带数据

    def __post_init__(self):
        if self.support_lines is None:
            self.support_lines = [13, 21, 34, 55, 89, 144]
        if self.bs is None:
            self.bs = {}

    def update_all_bs(self,
                      curr_index: int,
                      curr_min1: pd.Series,  # 当前bar 1分钟数据
                      ):
        """更新全部len的布林带数据"""
        if not len(self.bs):
            return

        for len_ in self.all_lines:
            # 每次都更新的值
            bl_mid = curr_min1[f'boll_mid_{self.min1_ma_type}_{len_}']
            bl_std = curr_min1[f'boll_std_{self.min1_ma_type}_{len_}']
            # 遍历全部length，产生各length信号
            self.bs[len_].update(
                ref=curr_min1[self.c_price_type],
                mid=bl_mid,  # 中值
                std=bl_std,  # std
                curr_index=curr_index, )

    @property
    def all_support_line(self) -> Dict[int, BlStatus]:
        """所有的有效强支撑线"""
        result = {}
        for len_ in self.support_lines:
            if self.bs[len_].is_support_line != Direction.NONE:
                result[len_] = self.bs[len_]
        return result

    @property
    def main_support_direction(self) -> Tuple[Direction, int]:
        """所有的有效支撑线中 选出主交易方向"""
        d = Direction.NONE
        l = None
        max_ratio = 0.99
        for len_ in self.support_lines:
            if self.bs[len_].mid_past_ratio > max_ratio:
                d = self.bs[len_].price_mid
                l = len_
                max_ratio = self.bs[len_].mid_past_ratio
        return d, l

    def useful_support_line(self):
        """有用的强支撑线"""
        # todo 大周期短于小周期，无效

    # @property
    # def smallest_support_line(self) -> Optional[int]:
    #     """最小周期的有效强支撑线"""

    def has_support_line(self,
                         direction: Direction,
                         ) -> Optional[int]:  # 支撑线的length
        """是否有有效强支撑线"""
        assert direction != Direction.NONE
        for len_ in self.support_lines:
            if direction == self.bs[len_].is_support_line:
                return len_
        return

    def cross_last_support_line(self,
                                target_direction: Direction,  # 支撑线要突破的方向
                                ) -> Optional[int]:  # 返回突破的支撑线的length
        """突破最后一条有效强支撑线"""
        assert target_direction != Direction.NONE
        flag = None
        for len_ in self.support_lines:
            # 任意一条有效强支撑线活着就False
            if Direction.is_opposite(target_direction,
                                     self.bs[len_].is_support_line):
                return
            # 任意一条满足，标记一下先
            if self.bs[len_].cross_support_line == target_direction:
                flag = len_
        return flag


@dataclass
class BlDevStatus:
    """
    记录 布林带偏移 状态
    """
    tp_range_trigger1: float = 4.0  # 止盈触发1 主（触发要改变状态的）
    tp_range_trigger2_p: float = 0.75  # 止盈触发2 相对止盈触发1的比值
    tp_range_draw_down2_p: float = 0.6  # 回撤最大range的多少 止盈2 比值
    sl_range_trigger1_p: float = 0.35  # 止损触发 相对止盈触发1的比值
    sl_range_draw_down_p: float = 0.9  # 回撤最大range的多少 止损 比值
    dev_max: float = -10  # 波段最大dev
    dev_min: float = 10  # 波段最小dev
    base_direction: Direction = None  # 基于的方向
    enable_other_trigger: bool = True  # 是否允许主trigger以外的使用

    # 需动态更新的
    range: float = field(init=False)  # 波段最大range
    tp1_triggered: Direction = Direction.NONE  # 触发了止盈1  返回止盈订单方向
    tp2_triggered: Direction = Direction.NONE  # 触发了止盈2  返回止盈订单方向
    need_tp2: Direction = Direction.NONE  # 需要止盈  返回止盈订单方向
    sl_triggered: Direction = Direction.NONE  # 触发了止损  返回止损订单方向
    need_sl: Direction = Direction.NONE  # 需要止损  返回止损订单方向

    def __post_init__(self):
        self.range = self.dev_max - self.dev_min

    def update(self, realtime_dev: float):
        """根据实时dev更新状态"""
        self.dev_max = max(realtime_dev, self.dev_max)
        self.dev_min = min(realtime_dev, self.dev_min)
        self.range = range_ = self.dev_max - self.dev_min

        # 触发 主门限
        if self.range > self.tp_range_trigger1:
            self.tp1_triggered = (Direction.LONG if realtime_dev == self.dev_max
                                  else Direction.SHORT)

            if not self.enable_other_trigger:  # 只更新高低点
                if realtime_dev == self.dev_max:
                    self.dev_min = realtime_dev  # 更新最小值
                    self.base_direction = Direction.SHORT
                    self.range = 0
                else:  # 到达低点
                    self.dev_max = realtime_dev  # 更新最大值
                    self.base_direction = Direction.LONG
                    self.range = 0

        if not self.base_direction or not self.enable_other_trigger or self.range == 0:
            return

        # 回撤 比值
        draw_down = ((self.dev_max - realtime_dev) / self.range
                     if self.base_direction == Direction.LONG else
                     (realtime_dev - self.dev_min) / self.range)

        if self.range > self.tp_range_trigger1 * self.tp_range_trigger2_p:
            self.tp2_triggered = (Direction.LONG if realtime_dev > self.dev_min
                                  else Direction.SHORT)  #

        if self.tp2_triggered != Direction.NONE:  # 曾经赋值过就OK
            self.need_tp2 = (self.base_direction if draw_down > self.tp_range_draw_down2_p
                             else Direction.NONE)

        # 触发止损
        if self.range > self.tp_range_trigger1 * self.sl_range_trigger1_p:
            self.sl_triggered = (Direction.LONG if realtime_dev > self.dev_min
                                 else Direction.SHORT)  #

        if self.sl_triggered != Direction.NONE:  # 曾经赋值过就OK
            self.need_sl = (self.base_direction if draw_down > self.sl_range_draw_down_p
                            else Direction.NONE)

    def change_predict_direction(self, new_direction: Direction):
        """外部来改变预测方向"""
        assert new_direction != Direction.NONE
        self.base_direction = new_direction
        if new_direction == Direction.LONG:
            self.dev_max = self.dev_min
        else:
            self.dev_min = self.dev_max


@dataclass
class PriceStatus:
    """
    记录 价格运动 状态
    """
    open_price: float  # 起始价格
    ref: float = None  # 参考价格/极值点价格/止损价格
    sl_range_threshold_p: float = 0  # 回撤时，之前反弹最低幅度需要多少 %
    sl_range_draw_down_p: float = None  # 回撤最大range的多少 止损 比值
    base_direction: Direction = Direction.NONE  # 基于的方向
    price_max: float = field(init=False)  # 最大
    last_extreme: Direction = Direction.NONE  # 最后的极值点
    last_dynamic_extreme: Direction = Direction.LONG  # 最后的动态极值点 假设开始long
    price_min: float = field(init=False)  # 最小
    range_price_max: float = field(init=False)  # 波段最大
    range_price_min: float = field(init=False)  # 波段最小
    dynamic_threshold_p = 0.5  # 门限 0.5%
    dynamic_max: float = field(init=False)  # 动态的最近高点 类似zigzag过门限
    dynamic_min: float = field(init=False)  # 动态的最近低点
    dynamic_mode: bool = True  # 仅更新动态

    # 需动态更新的
    range: float = field(init=False)  # 波段最大range

    def __post_init__(self):
        self.price_max = self.price_min = self.open_price
        self.dynamic_max = self.dynamic_min = self.open_price
        self.range_price_max = self.range_price_min = self.open_price
        self.range = 0

    def update(self, realtime_price: float):
        """根据实时price更新状态"""
        assert self.base_direction != Direction.NONE
        if realtime_price > self.price_max:
            self.price_max = realtime_price
            self.last_extreme = Direction.LONG  # 最后是高点的极值点
            if self.base_direction == Direction.SHORT:
                self.range_price_min = realtime_price
        elif realtime_price < self.price_min:
            self.price_min = realtime_price
            self.last_extreme = Direction.SHORT
            if self.base_direction == Direction.LONG:
                self.range_price_max = realtime_price  # 做多，新低时清空波段高点

        # 极值点后的 小范围的
        self.range_price_max = max(realtime_price, self.range_price_max)
        self.range_price_min = min(realtime_price, self.range_price_min)

    def update_dynamic(self, realtime_price: float):
        """根据实时price更新dynamic状态"""

        # 动态的高低点
        if self.last_dynamic_extreme == Direction.LONG:
            self.dynamic_max = max(realtime_price, self.dynamic_max)
            if ((self.dynamic_max - realtime_price) / self.dynamic_max
                    > self.dynamic_threshold_p / 100):  # 触发
                self.dynamic_min = realtime_price
                self.last_dynamic_extreme = Direction.SHORT
        else:  # Direction.SHORT 之前是低点
            self.dynamic_min = min(realtime_price, self.dynamic_min)
            if ((realtime_price - self.dynamic_min) / self.dynamic_min
                    > self.dynamic_threshold_p / 100):  # 触发
                self.dynamic_max = realtime_price
                self.last_dynamic_extreme = Direction.LONG

    @property
    def max_floating_loss(self):
        """最大浮亏"""
        loss = (self.price_max - self.open_price if self.base_direction == Direction.SHORT
                else self.open_price - self.price_min)
        return loss / self.open_price

    def reached_range_draw_down(self, now_price: float) -> bool:
        """
        到达波段回撤门限
        """
        if self.base_direction == Direction.LONG:
            # 从最低点开始反弹了多少
            if self.last_extreme == Direction.LONG:
                range_ = self.price_max - self.price_min
                draw_down = self.price_max - now_price  # 从最高点回落了多少
            elif self.last_extreme == Direction.SHORT:
                range_ = self.range_price_max - self.price_min
                draw_down = self.range_price_max - now_price  # 又回落了多少
            else:
                return False

            # if range_ / self.price_min < self.sl_range_threshold_p / 100:  # 超过门限
            #     return False

            # if now_price == self.price_min:
            #     print(f'最低点')
            #     return True

        else:  # SHORT
            if self.last_extreme == Direction.SHORT:
                range_ = self.price_max - self.price_min
                draw_down = now_price - self.price_min
            elif self.last_extreme == Direction.LONG:
                range_ = self.price_max - self.range_price_min
                draw_down = now_price - self.range_price_min
            else:
                return False

            # if range_ / self.price_max < self.sl_range_threshold_p / 100:
            #     return False

        return draw_down / range_ > self.sl_range_draw_down_p

# @dataclass
# class MaStatus:
#     """
#     记录 均线与价格相互的状态，交叉后重新更新
#     """
#     # 均线与close价格的相对位置，Long==close在上
#     position: Direction = Direction.NONE
#     start_index: int = 0
#     touched: int = 0  # 触碰到的次数
#     past: int = 0  # 已经过了几个bar了
#     touch_past: int = 0  # 距离上次touch 过了几个bar了
#     value: float = 0  # 均线的值
#
#     def update(self, touched: int, past: int, touch_past: int,
#                position: Direction = None, start_index: int = None,
#                value: float = 0):
#         """更新"""
#         self.touched: int = touched
#         self.past: int = past
#         self.touch_past: int = touch_past
#         if position:
#             self.position = position
#         if start_index:
#             self.start_index = start_index
#         if value:
#             self.value = value
#
#
# @dataclass
# class BollingStatus:
#     """
#     记录 布林带与价格相互的状态，交叉后重新更新
#     """
#     len_: int  # 计算周期
#     # Long==上轨上  SHORT==下轨下
#     position: Direction = Direction.NONE
#     prior_position: Direction = Direction.NONE
#     start_index: int = 0  # 突破时的布林带对应bar
#     start_value: float = 0  # 突破时的布林带对应值
#     past: int = 0  # 已经过了几个bar了
#
#     touched: int = 0  # 触碰到的次数
#     touch_past: int = 0  # 距离上次touch 过了几个bar了
#
#     def update(self, touched: int, past: int, touch_past: int,
#                position: Direction = None, prior_position: Direction = None,
#                start_index: int = None, value: float = 0):
#         """更新"""
#         self.touched: int = touched
#         self.past: int = past
#         self.touch_past: int = touch_past
#         if position:
#             self.position = position
#         if prior_position:
#             self.prior_position = prior_position
#         if start_index:
#             self.start_index = start_index
#         if value:
#             self.value = value
#
#     @property
#     def is_out_bl_line(self):
#         """在布林带外"""
#         return self.position is not Direction.NONE
