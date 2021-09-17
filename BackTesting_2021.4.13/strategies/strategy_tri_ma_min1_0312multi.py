#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略 新3均线 基于1分钟 不限制开仓 趋势反向开仓 不带日内平仓 带动态止盈
# 进程 出场都用1分钟

# Author: Tayii
# Data : 2021/03/05
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
import math

from config import BACK_TESTING_RESULT_DIR
from datahub.fmt import SimuTrade
from constant import Direction, OpenMax
from datahub.indicator import IndicatorCalc as Ind
from dc.config import DataSourceConfig
from utility import catch_except, timeit
from strategies.template import StrategyTemplate, TradeParams, OpenCloseParams
from backtester.template import BackTestingDataSetting
from constant import Sec


def set_bt_data_setting(data_source: DataSourceConfig,  # 数据源配置
                        new_bar_interval: int,  # 新bar 周期
                        indicators: Dict,  # 计算的指标
                        ) -> BackTestingDataSetting:
    """设置 多周期dataset继承类"""
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


bolling_line_list = range(10, 31, 5)  # 布林带length选择范围
min1_mp_list = [8, 13, 21, 34, 55, 89, 144, 233, 377, 610]  # 均值周期选择范围  # 5
min1_ma_comb = [(a, b, c) for a in min1_mp_list for b in min1_mp_list
                for c in min1_mp_list if a < b < c]  # 均值周期选择范围组合

min1_indicators = {}  # 需要预处理的数据指标
# 加 批量指标
for i in min1_mp_list:
    min1_indicators[f'ema_{i}'] = eval(f'lambda df: Ind.ema(df, {i})')
for j in bolling_line_list:
    min1_indicators[f'bolling_sma_{j}'] = eval(f'lambda df: Ind.boll(df, n={j})')

# 超参数
hyper_parameter = [{"min1_ma_periods": min1_ma_periods,  # 一分钟线 4个周期
                    'min1_ma_type': min1_ma_type,
                    'min1_bl_len': min1_bl_len,
                    'min1_bl_dev': min1_bl_dev,  # 1分钟布林带 dev
                    'min1_bolling_ma_type': min1_bolling_ma_type,
                    # 进场
                    'in_price_type': in_price_type,  # 开仓判断时用什么的价格
                    'in_ma_patten': in_ma_patten,  # 开仓判断时 均线图形正相关还是负相关
                    # 出场
                    # 'take_profit_type': take_profit_type,  # 止盈方式
                    # 'out_price_type_bl': out_price_type_bl,  # 布林带出场判断时用什么的价格
                    'close_types': close_types,  # 触发平仓类型
                    # "stop_loss_type": stop_loss_type,
                    'out_price_type_fixed': out_price_type_fixed,  # 固定止损出场判断时用什么的价格
                    # 'stop_loss_fixed_p': stop_loss_fixed_p,  # 固定止损比例
                    'take_profit_type': take_profit_type,  # 止盈类型
                    'take_profit_trigger_p': take_profit_trigger_p,  # 止盈 trigger值 %, 0.1=0.1%
                    }
                   #
                   for min1_ma_periods in [(34, 89, 233), (21, 55, 144),
                                           (55, 144, 377), (55, 144, 610),
                                           (55, 233, 610), (8, 21, 55), ]  # min1_ma_comb
                   for min1_ma_type in ['ema', ]
                   for min1_bl_len in bolling_line_list
                   for min1_bl_dev in [round(x, 2) for x in (list(numpy.arange(2.81, 0.8, -0.1))
                                                             + list(numpy.arange(-0.8, -2.81, -0.1)))]
                   for min1_bolling_ma_type in ['ema', 'sma'][1:]
                   # 进场
                   for in_price_type in ['extreme', 'Close'][1:]  # 进场价格选择
                   for in_ma_patten in ['positive_correlation', 'negative_correlation'][:]
                   # 出场
                   # for take_profit_type in ['boll_Min1', ]
                   # for out_price_type_bl in ['extreme', 'Close'][1:]  # 布林带出场判断时用什么的价格
                   for close_types in [['opposite_direction', 'day_end'][:1], ]
                   # for stop_loss_type in ['fixed', 'boll_Min1'][:1]
                   for out_price_type_fixed in ['extreme', 'Close'][1:]  # 固定止损出场判断时用什么的价格
                   # for stop_loss_fixed_p in numpy.arange(0.2, 0.5, 0.05)  # 固定止损比例)
                   for take_profit_type in ['trigger', ]
                   for take_profit_trigger_p in numpy.arange(0.2, 0.71, 0.05)
                   ]


#
# def select_hyper_parameter(plan_name: str) -> list:
#     """选中的部分超参数 一般用作二次回测"""
#     columns = list(hyper_parameter[0].keys())
#     total_csv_filepath = path.join(BACK_TESTING_RESULT_DIR,
#                                    f'{plan_name}', 'total_result.csv')
#     df = pd.read_csv(total_csv_filepath)
#     df = df[df['earnings'] > 0]
#     df = df[columns]
#     source: dict = df.to_dict('records')
#     ret = []
#     # 对原始数据进行处理
#     for s in source:
#         s['min1_ma_periods'] = eval(s['min1_ma_periods'])
#         s['bl_for_long_short'] = eval(s['bl_for_long_short'])
#
#     return ret


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
            },  # self.data_sets

            # 回测需要的超参数 list
            hyper_parameter=hyper_parameter[:],

            # 交易参数 开平仓等
            trade_params=TradeParams(
                symbol=data_source.symbol[:2],
                init_money=10000,  # 初始金额 USD
                max_lots=10,
                once_lots=1,
                open_max=OpenMax.LONG_n_OR_SHORT_n,  # 开仓无限制
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
        min1 = used_data['min1'].iloc[0]

        # 回测超参数 阈值等 -----------------------------
        min1_ma_periods = params.get('min1_ma_periods', )
        min1_ma_type = params.get('min1_ma_type')
        min1_bl_len: int = params.get('min1_bl_len')
        min1_bl_dev: float = params.get('min1_bl_dev')  # 1分钟布林带
        min1_bolling_ma_type = params.get('min1_bolling_ma_type')
        # 进场
        in_price_type = params.get('in_price_type', )  # 开仓判断时用什么的价格
        in_ma_patten = params.get('in_ma_patten', )  #
        # 出场
        # take_profit_type = params.get('take_profit_type', )  # 止盈方式
        # out_price_type_bl = params.get('out_price_type_bl', )  # 布林带出场判断时用什么的价格
        close_types = params.get('close_types', )  # 平仓类型
        stop_loss_type = params.get('stop_loss_type', None)  # 止损判断时用什么的价格
        out_price_type_fixed = params.get('out_price_type_fixed', )  # 固定止损出场判断时用什么的价格
        stop_loss_fixed_p = params.get('stop_loss_fixed_p', None)  # 固定止损比例
        take_profit_type = params.get('take_profit_type', )  # 止赢类型
        take_profit_trigger_p = params.get('take_profit_trigger_p', )  # 止盈 trigger值 %, 0.1=0.1%

        # 交易参数
        trade_p: TradeParams = self.trade_params  # 交易参数

        # 具体数据
        try:
            # 1分钟 MA的数值 转成通用格式
            min1_ma = {}
            for t, p in zip(('I', 'II', 'III'), min1_ma_periods):
                min1_ma[t] = min1[f'{min1_ma_type}_{p}']
                if math.isnan(min1_ma[t]):
                    return working_trades  # 数据未完全生成，退出这轮

            min1_bolling: Dict[str, float] = {
                'mid': min1[f'boll_mid_{min1_bolling_ma_type}_{min1_bl_len}'],
                'std': min1[f'boll_std_{min1_bolling_ma_type}_{min1_bl_len}'],
            }
            min1_bolling['up'] = min1_bolling['mid'] + min1_bl_dev * min1_bolling['std']
            min1_bolling['down'] = min1_bolling['mid'] - min1_bl_dev * min1_bolling['std']

        except Exception as e:
            self.log(f'data err: {e}')

        # 当前处理的bar的index
        curr_index = used_data['min1'].index[-1]

        # self.log(f'curr_index = {curr_index}  {used_data["min1"]["datetime"]}')

        # 具体逻辑 =============================

        def multi_ma_patten(minute_x_ma: dict,  # 哪个分钟周期的均线
                            ) -> Direction:
            """多均线 形态 """
            if (minute_x_ma['II'] > minute_x_ma['III']
                    and minute_x_ma['I'] > minute_x_ma['III']):
                return (Direction.LONG if in_ma_patten == 'positive_correlation'
                        else Direction.SHORT)  # 正向时多头排列 反向时空头排列
            # 所有短期均线MA < 最长期的
            elif (minute_x_ma['II'] < minute_x_ma['III']
                  and minute_x_ma['I'] < minute_x_ma['III']):
                return (Direction.SHORT if in_ma_patten == 'positive_correlation'
                        else Direction.LONG)  # 正向时空头排列 反向是多头排列
            else:
                return Direction.NONE

        def calc_open_direction() -> OpenCloseParams:
            """判断开仓方向"""
            # 根据当前均线排列，分开考虑
            multi_ma_direction = multi_ma_patten(min1_ma)

            # 均线多头排列 开多判断
            if multi_ma_direction == Direction.LONG:
                # 需要open在布林带上轨下方，且实时价格向上破轨道
                used_price_ = min1.Last if in_price_type == 'Close' else min1.High
                # ↓open这里约等同前bar收盘价
                if used_price_ > min1_bolling['up'] > min1.Open:
                    return OpenCloseParams(Direction.LONG,
                                           type=f'trend & cross up up-line',
                                           price=(min1.Last if in_price_type == 'Close'
                                                  else min1_bolling['up']))

            # 均线空头排列 开空判断
            elif multi_ma_direction == Direction.SHORT:
                # 需要open在布林带下轨上方，且实时价格向下破轨道
                used_price_ = min1.Last if in_price_type == 'Close' else min1.Low
                # ↓open这里约等同前bar收盘价
                if used_price_ < min1_bolling['down'] < min1.Open:
                    return OpenCloseParams(Direction.SHORT,
                                           type='trend & cross down down-line',
                                           price=(min1.Last if in_price_type == 'Close'
                                                  else min1_bolling['down']))

            # 震荡排列 多空一起判断
            # else:
            #     # 先判断开多
            #     # 实时上破布林带下轨
            #     if min1.Open < min1_bolling['down']:
            #         used_price_ = min1.Last if in_price_type == 'Close' else min1.High
            #         if min1_bolling['mid'] > used_price_ > min1_bolling['down']:
            #             return OpenCloseParams(Direction.LONG,
            #                                    type='congestion & cross up down-line',
            #                                    price=(min1.Last if in_price_type == 'Close'
            #                                           else min1_bolling['down']))
            #     # 判断开空
            #     # 实时下破布林带上轨
            #     elif min1.Open > min1_bolling['up']:
            #         used_price_ = min1.Last if in_price_type == 'Close' else min1.Low
            #         if min1_bolling['mid'] < used_price_ < min1_bolling['up']:
            #             return OpenCloseParams(Direction.SHORT,
            #                                    type='congestion & cross down up-line',
            #                                    price=(min1.Last if in_price_type == 'Close'
            #                                           else min1_bolling['up']))
            # 都不满足开仓
            return OpenCloseParams()

        # 开仓逻辑开始 ======================================
        after_open_processed_trades, result = [], []

        def _process_open(trade_: SimuTrade):
            """开仓处理"""
            try:
                # 没开仓 且 满足开仓 用SimTrade.set_open
                trade_.set_open(direction_=open_signal_.direction,
                                bar=curr_index,  # 当前bar
                                price=open_signal_.price,  # 当前价
                                amount=trade_p.once_lots,  # 开仓量
                                datetime_=min1['datetime'],  # 开仓时间
                                open_type=open_signal_.type
                                )
            except Exception as e:
                self.log(f'work err: {e}', logging.ERROR, exc_info=True)
            finally:
                # 返回全部回测trade 不管是否成功，异常
                return trade_

        # 获取开仓信号 开仓信号可以统一获取）
        open_signal_ = calc_open_direction()  # 开仓信息

        # 开仓处理 遍历各工单(未开仓/已开仓未平仓）
        for a_trade in working_trades:
            if a_trade.waiting_open:
                if open_signal_.direction != Direction.NONE:  # 有信号
                    a_trade = _process_open(a_trade)  # 开仓 处理
            else:
                # 已开仓的 只更新最大浮盈
                a_trade.record_extreme(high=min1.High, low=min1.Low)
            # 所有的工单都保存起来
            after_open_processed_trades.append(a_trade)

        # 获取平仓信号  ==============================================

        def single_get_close_signal(trade_: SimuTrade, ) -> OpenCloseParams:
            """单工单 获取平仓信号"""

            # 反转平仓==用开仓信号反向进行平仓
            if 'opposite_direction' in close_types:
                # 现在开仓信号与订单的相反  # 用反向开仓的价格平仓
                if (open_signal_.direction != Direction.NONE
                        and open_signal_.direction != trade_.direction):
                    return OpenCloseParams(direction=trade_.direction,
                                           type='opposite_open_direction',
                                           price=open_signal_.price)

                    # 日内平仓
            if 'day_end' in close_types and trade_.waiting_close:
                if min1.datetime.time() == datetime.time(hour=17):
                    return OpenCloseParams(direction=trade_.direction,
                                           type='day_end',
                                           price=min1.Open)  # 最后一分钟开盘价平仓

            # 止损平仓 ==========================================

            if stop_loss_type and stop_loss_type == 'fixed':  # 固定止损
                if trade_.direction == Direction.LONG:
                    # 固定止损的触发价，==开盘价*（1-回落百分比）
                    sl_triggle_price = trade_.open_price * (1 - stop_loss_fixed_p / 100)
                    # 判断用的价格
                    used_price_ = (min1.Last if out_price_type_fixed == 'Close' else min1.Low)
                    if used_price_ <= sl_triggle_price:
                        return OpenCloseParams(direction=trade_.direction,
                                               type='fixed_stop_loss',
                                               price=(used_price_
                                                      if out_price_type_fixed == 'Close'
                                                      else sl_triggle_price), )
                elif trade_.direction == Direction.SHORT:  # 平空
                    sl_triggle_price = trade_.open_price * (1 + stop_loss_fixed_p / 100)
                    used_price_ = (min1.Last if out_price_type_fixed == 'Close' else min1.High)
                    if used_price_ >= sl_triggle_price:
                        return OpenCloseParams(direction=trade_.direction,
                                               type='fixed_stop_loss',
                                               price=(used_price_
                                                      if out_price_type_fixed == 'Close'
                                                      else sl_triggle_price), )

            # 止盈平仓 ==========================================

            if take_profit_type and take_profit_type == 'trigger':  # 移动止盈
                if trade_.direction == Direction.LONG:
                    # 移动止盈的触发价，==最高价*（1-回落百分比）
                    tp_trigger_price = trade_.highest_v * (1 - take_profit_trigger_p / 100)
                    if min1.Last <= tp_trigger_price:
                        return OpenCloseParams(direction=trade_.direction,
                                               type='trigger_take_profit',
                                               price=tp_trigger_price, )
                elif trade_.direction == Direction.SHORT:  # 平空
                    tp_trigger_price = trade_.lowest_v * (1 + take_profit_trigger_p / 100)
                    if min1.Last >= tp_trigger_price:
                        return OpenCloseParams(direction=trade_.direction,
                                               type='trigger_take_profit',
                                               price=tp_trigger_price, )

            # 都不符合
            return OpenCloseParams()  # 不平仓

        # def batch_get_close_signal() -> OpenCloseParams:
        #     """批量 获取平仓信号"""

        # 平仓处理
        def _process_close(trade_: SimuTrade,
                           close_signal: OpenCloseParams,  # 平仓信号
                           ) -> SimuTrade:
            try:
                # 平仓
                if trade_.waiting_close:
                    self.close_trade(trade_,
                                     close_bar=curr_index,
                                     close_type=close_signal.type,
                                     close_price=close_signal.price,
                                     datetime_=min1['datetime'])
            except Exception as e:
                self.log(f'work err: {e} {trade_}', logging.ERROR, exc_info=True)
            finally:
                # 返回全部回测trade 不管是否成功，异常
                return trade_

        # 全部订单 逐一判断平仓
        for a_trade in after_open_processed_trades:
            if a_trade.waiting_close:
                close_signal_ = single_get_close_signal(a_trade)
                if close_signal_.direction == a_trade.direction:  # 是这个工单需要的平仓方向
                    a_trade = _process_close(a_trade, close_signal_)
            # 收集回来
            result.append(a_trade)

        if result is None or len(result) == 0:
            pass
        return result  # body()
