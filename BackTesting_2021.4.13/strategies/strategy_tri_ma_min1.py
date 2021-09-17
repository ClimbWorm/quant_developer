#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略 三均线  只基于1分钟线 多周期调

# Author: Tayii
# Data : 2021/02/23
# ----------------------------------------------------
# from typing import Any, Callable#
# from datahub import BarGenerator, ArrayManager, TickData, BarData, OrderData, TradeData, StopOrder
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union
import datetime
import pandas as pd
import logging
from os import path
import numpy

from config import BACK_TESTING_RESULT_DIR
from datahub.fmt import SimuTrade
from constant import Direction, OpenMax
from datahub.indicator import IndicatorCalc as Ind
from dc.config import DataSourceConfig
from utility import catch_except
from strategies.template import StrategyTemplate, TradeParams
from backtester.template import BackTestingDataSetting
from constant import Sec


def set_bt_data_setting(data_source: DataSourceConfig,  # 数据源配置
                        new_bar_interval: int,  # 新bar 周期
                        indicators: Dict,  # 计算的指标
                        ) -> BackTestingDataSetting:
    """设置 多dataset继承类"""
    return BackTestingDataSetting(
        data_source=data_source,  # 数据源 配置文件名
        day_open=8.5,  # 每天开盘的时间 小时计 （可选）
        new_bar_day_open=datetime.time(8),  # 新bar 日盘开盘时间
        new_bar_night_open=datetime.time(17),  # 新bar 夜盘开盘时间
        # 策略需要的数据（列名）
        need_columns=['datetime', 'Open', 'High', 'Low', 'Last',
                      'Volume', 'timestamp'],
        new_bar_interval=new_bar_interval,
        indicators=indicators,
    )


ma_type_list = ['ema', 'sma']
bolling_line_list = range(10, 31, 5)  # 布林带length选择范围
min1_mp_list = [21, 34, 55, 89, 144, 233, 377, 610]  # 均值周期选择范围
bolling_dev = numpy.arange(0.5, 2.1, 0.25)  # 布林带 偏差倍数
min1_ma_comb = [(a, b, c) for a in min1_mp_list for b in min1_mp_list
                for c in min1_mp_list if a < b < c]  # 均值周期选择范围组合
min1_indicators = {}  # 需要预处理的数据指标
# 加 批量指标
for i in min1_mp_list:
    min1_indicators[f'ema_{i}'] = eval(f'lambda df: Ind.ema(df, {i})')
for j in bolling_line_list:
    min1_indicators[f'bolling_sma_{j}'] = eval(f'lambda df: Ind.boll(df, n={j})')

min5_indicators = {}  # 需要预处理的数据指标
# 加 批量指标
for j in bolling_line_list:
    min5_indicators[f'bolling_ema_{j}'] = eval(f'lambda df: Ind.boll(df, n={j}, ma_type="ema")')

# 超参数
hyper_parameter = [{"min1_ma_periods": min1_ma_periods,  # 一分钟线 4个周期
                    'min1_ma_type': min1_ma_type,
                    'min1_bolling_len': min1_bolling_len,
                    'min1_bolling_dev': min1_bolling_dev,
                    'min1_bolling_ma_type': min1_bolling_ma_type,
                    'min5_bolling_len': min5_bolling_len,
                    'min5_tp_bolling_dev': min5_tp_bolling_dev,
                    'min5_sl_bolling_dev': min5_sl_bolling_dev,
                    'min5_bolling_ma_type': min5_bolling_ma_type,
                    "bl_for_long_short": bl_for_long_short,  # 开仓上穿哪条线
                    # "bl_for_short": bl_for_short,
                    'decide_open_type': decide_open_type,  # 开仓判断时用什么的价格
                    'decide_open_ma_patten': decide_open_ma_patten,  # 开仓判断时 均线图形正相关还是负相关
                    "stop_loss_type": stop_loss_type,
                    'take_profit_type': take_profit_type,
                    }  # 止损方式
                   for min1_ma_periods in [(21, 55, 144), ]  # min1_ma_comb
                   for min1_ma_type in ['ema', ]
                   for min1_bolling_len in bolling_line_list[:-2]
                   for min1_bolling_dev in bolling_dev[1:]
                   for min1_bolling_ma_type in ['sma', ]
                   for min5_bolling_len in bolling_line_list[:-2]
                   for min5_tp_bolling_dev in bolling_dev[2:]
                   for min5_sl_bolling_dev in bolling_dev[2:]
                   for min5_bolling_ma_type in ['ema', ]
                   for bl_for_long_short in [("up", "down"), ("mid", "mid"), ("down", "up")]
                   for decide_open_type in ['extreme', 'Close'][1:]
                   for decide_open_ma_patten in ['positive_correlation', 'negative_correlation'][:]
                   for stop_loss_type in ['boll_II', 'boll_I', 'bar'][:1]
                   for take_profit_type in ['boll_II', ]
                   ]


def select_hyper_parameter(plan_name: str) -> list:
    """选中的部分超参数 一般用作二次回测"""
    columns = list(hyper_parameter[0].keys())
    total_csv_filepath = path.join(BACK_TESTING_RESULT_DIR,
                                   f'{plan_name}', 'total_result.csv')
    df = pd.read_csv(total_csv_filepath)
    df = df[df['earnings'] > 0]
    df = df[columns]
    source: dict = df.to_dict('records')
    ret = []
    # 对原始数据进行处理
    for s in source:
        s['min1_ma_periods'] = eval(s['min1_ma_periods'])
        s['bl_for_long_short'] = eval(s['bl_for_long_short'])

    return ret


class Strategy(StrategyTemplate):
    """
    策略 类名必须Strategy
    """

    def __init__(self,
                 data_source: DataSourceConfig,  # 数据源配置
                 name: str = None,
                 ):
        # 在下面 输入 对应本策略 定制的各类参数 =======================================

        StrategyTemplate.__init__(
            self,
            name=name or self.__class__.__name__,
            need_bars_once=1,  # 一次切片回测需要最近1个bar数据

            # 回测用数据配置文件们
            data_sets={
                # 合成1分钟的数据及指标的配置
                'min1': set_bt_data_setting(
                    data_source=data_source,
                    new_bar_interval=Sec.MIN1.value,  # 新bar 周期
                    indicators=min1_indicators,
                ),
                # 合成5分钟的数据及指标的配置
                'min5': set_bt_data_setting(
                    data_source=data_source,
                    new_bar_interval=Sec.MIN5.value,  # 新bar 周期
                    indicators=min5_indicators,
                ),
            },  # self.data_sets

            # 回测需要的超参数 list
            hyper_parameter=hyper_parameter[:],
            # 交易参数 开平仓等
            trade_params=TradeParams(
                symbol=data_source.symbol[:2],
                init_money=10000,  # 初始金额 USD
                max_lots=10,
                once_lots=1,
                open_max=OpenMax.LONG_1_OR_SHORT_1,  # 开仓限制
                # fee_rate= 0.1,  # 交易手续费 %（可选，优先于fee_amount）
                fee_amount=5.0,  # 交易手续费 固定量（可选）
                slippage=2,  # 交易滑点  tick数
            ),

            # 交易结果计算 参数
            result_paras={
            },

            save_data_result=True,  # 保存回测数据结果
            show_data_result=True,  # 显示回测数据结果
            save_trade_result=True,  # 保存回测交易结果

        )  # StrategyTemplate（）
        # self.import_path = f'strategies.{path.split(path.realpath(__file__))[1]}'

    @catch_except()
    def body(self,
             working_trades: list,  # 要回测的交易们 (未开仓/已开仓未平仓）
             used_data: Dict[str, pd.DataFrame],  # 已处理好的数据（不同周期的数据 都取最近n个切片）
             params: dict,  # 逻辑判断需要的各个参数 门限值
             ) -> List[SimuTrade]:  # 新的交易状态
        """策略运行 主程序"""

        # 切片数据  curr当前bar
        min1, min5 = used_data['min1'].iloc[0], used_data['min5'].iloc[0]

        # 回测超参数 阈值等 -----------------------------
        min1_ma_periods = params.get('min1_ma_periods', )
        min1_ma_type = params.get('min1_ma_type')
        min1_bolling_len = params.get('min1_bolling_len')
        min1_bolling_dev = params.get('min1_bolling_dev')
        min1_bolling_ma_type = params.get('min1_bolling_ma_type')
        min5_bolling_len = params.get('min5_bolling_len')
        min5_bolling_ma_type = params.get('min5_bolling_ma_type')
        min5_tp_bolling_dev = params.get('min5_tp_bolling_dev')  # 止盈的
        min5_sl_bolling_dev = params.get('min5_sl_bolling_dev')  # 止损的

        bl_for_long, bl_for_short = params.get('bl_for_long_short')
        if isinstance(bl_for_long, str):
            bl_for_long = [bl_for_long, ]
        if isinstance(bl_for_short, str):
            bl_for_short = [bl_for_short, ]
        decide_open_type = params.get('decide_open_type', )  # 开仓判断时用什么的价格
        decide_open_ma_patten = params.get('decide_open_ma_patten', )
        stop_loss_type = params.get('stop_loss_type', )  # 止损方式
        take_profit_type = params.get('take_profit_type', )  # 止盈方式

        # 交易参数
        trade_p: TradeParams = self.trade_params  # 交易参数

        # 参数本身取值范围
        ma_pt = ('I', 'II', 'III')  # I最短周期

        # 具体数据
        try:
            min1_ma = {}  # 1分钟 MA的数值
            for t, p in zip(ma_pt, min1_ma_periods):
                min1_ma[t] = min1[f'{min1_ma_type}_{p}']

            min1_bolling, min5_tp_bolling, min5_sl_bolling = {}, {}, {}
            min1_bolling['mid'] = min1[f'boll_mid_{min1_bolling_ma_type}_{min1_bolling_len}']
            min1_bolling['std'] = min1[f'boll_std_{min1_bolling_ma_type}_{min1_bolling_len}']
            min1_bolling['up'] = min1_bolling['mid'] + min1_bolling_dev * min1_bolling['std']
            min1_bolling['down'] = min1_bolling['mid'] - min1_bolling_dev * min1_bolling['std']

            min5_tp_bolling['mid'] = min5[f'boll_mid_{min5_bolling_ma_type}_{min5_bolling_len}']
            min5_tp_bolling['std'] = min5[f'boll_std_{min5_bolling_ma_type}_{min5_bolling_len}']

            min5_tp_bolling['up'] = min5_tp_bolling['mid'] + min5_tp_bolling_dev * min5_tp_bolling['std']
            min5_tp_bolling['down'] = min5_tp_bolling['mid'] - min5_tp_bolling_dev * min5_tp_bolling['std']

            min5_sl_bolling['up'] = min5_tp_bolling['mid'] + min5_sl_bolling_dev * min5_tp_bolling['std']
            min5_sl_bolling['down'] = min5_tp_bolling['mid'] - min5_sl_bolling_dev * min5_tp_bolling['std']

        except Exception as e:
            self.log(f'data err: {e}')

        # 当前处理的bar的index
        curr_index = used_data['min1'].index[-1]

        # self.log(f'curr_index = {curr_index}  {used_data["min1"]["datetime"]}')

        def _process(trade: SimuTrade) -> SimuTrade:
            """具体逻辑"""

            direction = Direction.NONE  # 交易方向

            def ma_patten_lone(minute_x_ma: dict) -> bool:
                """多均线 形态 是否多头排列"""
                return (minute_x_ma['I'] > minute_x_ma['III']
                        and minute_x_ma['II'] > minute_x_ma['III'])

            def ma_patten_short(minute_x_ma: dict) -> bool:
                """多均线 形态 是否空头排列"""
                return (minute_x_ma['I'] < minute_x_ma['III']
                        and minute_x_ma['II'] < minute_x_ma['III'])

            def open_long() -> Union[bool, Tuple[bool, Any]]:
                """开多判断"""
                # 短期均线MA > 最长期的
                if decide_open_ma_patten == 'positive_correlation':
                    if not ma_patten_lone(min1_ma):
                        return False
                elif decide_open_ma_patten == 'negative_correlation':
                    if not ma_patten_short(min1_ma):
                        return False

                # 1min 是阳线
                if min1.Last <= min1.Open:
                    # self.log(f'open_long 1min "是阳线" 不满足')
                    return False
                # open在轨道下方，且布林带实时价格向上破 中轨/下轨/上轨 （或组合）
                used_price_ = (min1.Last if decide_open_type == 'Close' else min1.High)
                for blt in bl_for_long:
                    if used_price_ > min1_bolling[blt] > min1.Open:  # open这里约等同前bar收盘价
                        # self.log(f'open_long 1min  **** 满足')
                        return True, min1.Last if decide_open_type == 'Close' else min1_bolling[blt]  # 穿过一条即可

                # self.log(f'open_long 1min "open在轨道下方，且布林带实时价格向上破 中轨/下轨/上轨 " 不满足')
                return False

            def open_short() -> Union[bool, Tuple[bool, Any]]:
                """开空判断"""
                # 短期均线MA < 最长期的
                if decide_open_ma_patten == 'positive_correlation':
                    if not ma_patten_short(min1_ma):
                        return False
                elif decide_open_ma_patten == 'negative_correlation':
                    if not ma_patten_lone(min1_ma):
                        return False

                # 1min
                # 是阴线
                if min1.Last >= min1.Open:
                    # self.log(f'open_short 1min 阴线" 不满足')
                    return False
                # open在轨道上方，且布林带实时价格向下破 中轨/下轨/上轨 （或组合）
                used_price_ = (min1.Last if decide_open_type == 'Close' else min1.Low)
                for blt in bl_for_short:
                    if used_price_ < min1_bolling[blt] < min1.Open:  # open这里约等同前bar收盘价
                        # self.log(f'open_short 1min **** 满足')
                        return True, min1.Last if decide_open_type == 'Close' else min1_bolling[blt]  # 穿过一条即可

                # self.log(f'open_short 1min open在轨道上方，且布林带实时价格向下破 中轨/下轨/上轨 不满足')
                return False

            def close_long() -> Union[bool, Tuple[bool, Any]]:
                """平多判断"""
                # 止盈  5分钟布林带上轨外
                used_price_ = (min1.Last if decide_open_type == 'Close' else min1.High)
                if used_price_ > min5_tp_bolling['up']:
                    return True, min1.Last if decide_open_type == 'Close' else min5_tp_bolling['up']

                # 止损
                p_ = {'bar': trade.stop_loss,  # bar low
                      'boll_I': min1_bolling['down'],
                      'boll_II': min5_sl_bolling['down'],
                      }
                used_price_ = (min1.Last if decide_open_type == 'Close' else min1.Low)
                if used_price_ < p_[stop_loss_type]:
                    return True, min1.Last if decide_open_type == 'Close' else p_[stop_loss_type]
                else:
                    return False

            def close_short() -> Union[bool, Tuple[bool, Any]]:
                """平空判断"""
                # 止盈  5分钟布林带下轨外
                used_price_ = (min1.Last if decide_open_type == 'Close' else min1.Low)
                if used_price_ < min5_tp_bolling['down']:
                    # self.log(f"close_short  不满足 {curr_min1.Last} < {bolling['curr']['II']['down']}")
                    return True, min1.Last if decide_open_type == 'Close' else min5_tp_bolling['down']

                # 止损
                p_ = {'bar': trade.stop_loss,  # bar high
                      'boll_I': min1_bolling['up'],
                      'boll_II': min5_sl_bolling['up'],
                      }
                used_price_ = (min1.Last if decide_open_type == 'Close' else min1.High)
                if used_price_ > p_[stop_loss_type]:
                    return True, min1.Last if decide_open_type == 'Close' else p_[stop_loss_type]
                else:
                    # self.log(f"close_short  不满足 curr_min1.Last{curr_min1.Last} > {p[stop_loss_type]}")
                    return False

            # 逻辑开始 ==================
            if trade.waiting_open:  # 开仓进行判断
                open_ret = open_long()
                if open_ret:
                    direction = Direction.OPEN_LONG  # 开多
                else:
                    open_ret = open_short()
                    if open_ret:
                        direction = Direction.OPEN_SHORT  # 开空

                # 满足开仓 用SimTrade.set_open
                if direction != Direction.NONE:
                    trade.set_open(direction,
                                   curr_index,  # 当前bar
                                   open_ret[1],  # 当前价
                                   amount=trade_p.once_lots,  # 开仓量
                                   datetime_=min1['datetime'],  # 开仓时间
                                   stop_loss=(min1['Low'] if direction == Direction.OPEN_LONG
                                              else min1['High'])
                                   )

                return trade

            else:  # 平仓进行判断
                # 更新最大浮盈
                trade.record_extreme(high=min1.High, low=min1.Low)

                close_ret = None
                # 多头平仓
                if trade.direction == Direction.OPEN_LONG:
                    close_ret = close_long()
                    if close_ret:
                        direction = Direction.CLOSE_LONG
                # 空头平仓
                elif trade.direction == Direction.OPEN_SHORT:
                    close_ret = close_short()
                    if close_ret:
                        direction = Direction.CLOSE_SHORT

                # 满足平仓 用SimTrade.set_close 数量==开仓数量
                if direction in [Direction.CLOSE_LONG, Direction.CLOSE_SHORT]:
                    trade.set_close(curr_index,  # 当前bar
                                    close_ret[1],  #
                                    min1['datetime'],  # 平仓时间
                                    )
                # 返回最后结果
                return trade

        # 遍历各交易(未开仓/已开仓未平仓）
        result = []
        for each_trade in working_trades:
            try:
                each_trade = _process(each_trade)
            except Exception as e:
                self.log(f'work err: {e}', logging.ERROR, exc_info=True)
            finally:
                result.append(each_trade)  # 返回全部回测trade 不管是否成功，异常

        return result
