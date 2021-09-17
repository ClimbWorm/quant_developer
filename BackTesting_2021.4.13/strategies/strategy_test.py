#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略 test

# Author: Tayii
# Data : 2020/12/02
# ----------------------------------------------------
# from typing import Any, Callable#
# from strategies.template import StrategyTemplate
# from datahub import BarGenerator, ArrayManager, TickData, BarData, OrderData, TradeData, StopOrder
from typing import List

import pandas as pd
import logging

from datahub.fmt import SimuTrade
from constant import Direction, OpenMax
from datahub.indicator import IndicatorCalc as Ind
from utility import catch_except
from strategies.template import StrategyTemplate


class Strategy(StrategyTemplate):
    """
    策略 类名必须Strategy
    """

    def __init__(self):
        StrategyTemplate.__init__(self)
        self.need_bars_once = 2  # 一次切片回测需要最近几个bar数据

        # 在下面 输入 对应本策略 定制的各类参数 =======================================

        # 策略需要的数据（列名）
        self.need_columns = ['datetime', 'Open', 'High', 'Low', 'Last',
                             'Volume', 'timestamp']
        # 需要预处理的数据指标
        self.indicators = {
            'day_extremum': lambda df: Ind.day_extremum(df),
            'zigzag': lambda df: Ind.zigzag(df, 0.2, mode='ratio'),
            'roc_6': lambda df: Ind.roc(df, 6),
            'rocp_6': lambda df: Ind.rocp(df, 6),
            'sma_8': lambda df: Ind.sma(df, 8),
            # 'std_8': lambda df: Ind.std(df, 8),
            # 'cmo_11': lambda df: Ind.cmo(df, 11),
            # 'plus_di_5': lambda df: Ind.plus_di(df, 5),
            # 'dx_9': lambda df: Ind.dx(df, 9),
            # 'atr_15': lambda df: Ind.atr(df, 15),
        }
        # 加 批量指标
        for i in range(5, 7):
            self.indicators[f'rsi_{i}'] = eval(f'lambda df: Ind.rsi(df, {i})')

        # 回测需要的超参数 list
        self.hyper_parameter = [{"rsi_open": rsi_open, "rsi_close": rsi_close, }
                                for rsi_open in range(75, 88, 31)
                                for rsi_close in [  15,]]

        # 交易参数 开平仓等
        self.trade_params = {
            'init_money': 200000,  # 初始金额 USD
            'fee_rate': 0.1,  # 交易手续费 %（可选，优先于fee_amount）
            # 'fee_amount': 0.0,  # 交易手续费 固定量（可选）
            'max_lots': 10,
            'once_lots': 5,
            'open_max': OpenMax.LONG_1_OR_SHORT_1,  # 开仓限制
            'take_profit': 5,
            'stop_loss': 6,
        }

        # 交易结果计算 参数
        self.result_paras = {
        }


    @catch_except()
    def body(self,
             working_trades: list,  # 要回测的交易们 (未开仓/已开仓未平仓）
             used_df: pd.DataFrame,  # 已处理好的数据（输入是策略需要的最近n个切片）
             paras: dict,  # 逻辑判断需要的各个参数 门限值
             ) -> List[SimuTrade]:  # 新的交易状态
        """策略运行 主程序"""

        # 回测参数 阈值
        rsi_open = paras.get('rsi_open', -1)

        # 切片数据  curr当前bar  prior前bar 。。。
        prior_bar, curr_bar = used_df.iloc[0], used_df.iloc[1]  # 最后一个当前bar
        # 当前处理的bar index
        curr_index = used_df.index[-1]

        def _process(trade: SimuTrade) -> SimuTrade:
            """具体逻辑"""
            if trade.waiting_open:
                # 开仓进行判断

                # 开多判断

                # 开空判断
                if rsi_open <= 0:
                    print(f'入参出错')

                elif curr_bar.rsi_5 > rsi_open:
                    # print(f'curr.rsi_5 {curr.rsi_5} > {rsi_open}, open')

                    # 满足开仓 用SimTrade.set_open
                    trade.set_open(Direction.LONG,  # 开多
                                   curr_index,  # 当前bar
                                   curr_bar['Last'],  # 当前价
                                   5,  # 开仓量
                                   curr_bar['datetime'],  # 开仓时间
                                   )

            # 平仓进行判断
            else:
                # print(f'平仓进行判断..........')
                rsi_close = paras.get('rsi_close', -1)
                if rsi_close <= 0:
                    print(f'入参出错')

                elif curr_bar.rsi_5 < rsi_close:
                    # print(f'curr.rsi_5  {curr.rsi_5} < {rsi_close}, close')

                    # 满足平仓 用SimTrade.set_close
                    trade.set_close(curr_index,  # 当前bar
                                    curr_bar['Last'],  # 当前价
                                    curr_bar['datetime'],  # 平仓时间
                                    )

            return trade

        # 遍历各交易(未开仓/已开仓未平仓）
        result = []
        for each_trade in working_trades:
            try:
                each_trade = _process(each_trade)
            except Exception as e:
                self.log(f'err: {e}', logging.ERROR, exc_info=True)
            finally:
                result.append(each_trade)  # 返回全部回测trade 不管是否成功，异常

        return result

#
#     parameters = []
#
#     variables = []
#
#     def __init__(self, engine: Any, strategy_name: str,
#                  e_symbol: str, setting: dict, ):
#         """"""
#         super().__init__(engine, strategy_name, e_symbol, setting)
#         self.name = 'test'
#         self.bg = BarGenerator(self.on_bar)
#         self.am = ArrayManager()
#
#     def on_init(self):
#         """
#         Callback when strategy is inited.
#         """
#         self.write_log(f"{self.name}策略初始化")
#
#     def on_start(self):
#         """
#         Callback when strategy is started.
#         """
#         self.write_log("策略启动")
#
#     def on_stop(self):
#         """
#         Callback when strategy is stopped.
#         """
#         self.write_log("策略停止")
#
#     def on_tick(self, tick: TickData):
#         """
#         Callback of new tick data update.
#         """
#         self.bg.update_tick(tick)
#
#     def on_bar(self, bar: BarData):
#         """
#         Callback of new bar data update.
#         """
#         print(f'TestStrategy on_bar()')
#         self.cancel_all()
#
#         am = self.am
#         am.update_bar(bar)
#         if not am.inited:
#             return
#
#     def on_order(self, order: OrderData):
#         """
#         Callback of new order data update.
#         """
#         pass
#
#     def on_trade(self, trade: TradeData):
#         """
#         Callback of new trade data update.
#         """
#         self.put_event()
#
#     def on_stop_order(self, stop_order: StopOrder):
#         """
#         Callback of stop order update.
#         """
#         pass
