#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略 三均线  跟原始进出场相反的

# Author: Tayii
# Data : 2020/12/02
# ----------------------------------------------------
# from typing import Any, Callable#
# from datahub import BarGenerator, ArrayManager, TickData, BarData, OrderData, TradeData, StopOrder
from dataclasses import dataclass
from typing import List, Dict
import datetime
import pandas as pd
import logging
from os import path

from datahub.fmt import SimuTrade
from constant import Direction, OpenMax
from datahub.indicator import IndicatorCalc as Ind
from dc.config import YMH21_1_SC, DataSourceConfig
from utility import catch_except
from strategies.template import StrategyTemplate, TradeParams
from backtester.template import BackTestingDataSetting
from constant import Sec


def set_BackTestingDataSetting(data_source: DataSourceConfig,  # 数据源配置
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
            need_bars_once=2,  # 一次切片回测需要最近2个bar数据

            # 回测用数据配置文件们
            data_sets={
                # 合成1分钟的数据及指标的配置
                'min1': set_BackTestingDataSetting(
                    data_source=data_source,
                    new_bar_interval=Sec.MIN1.value,  # 新bar 周期
                    indicators={  # 需要预处理的数据指标
                        'sma_bolling': lambda df: Ind.boll(df, 20, 2.0),
                        'ema_bolling': lambda df: Ind.boll(df, 20, 2.0, 'ema'),
                    },
                ),
                # 合成5分钟的数据及指标的配置
                'min5': set_BackTestingDataSetting(
                    data_source=data_source,
                    new_bar_interval=Sec.MIN5.value,  # 新bar 周期
                    indicators={  # 需要预处理的数据指标
                        'sma_5': lambda df: Ind.sma(df, 5),
                        'ema_5': lambda df: Ind.ema(df, 5),
                        'sma_10': lambda df: Ind.sma(df, 10),
                        'ema_10': lambda df: Ind.ema(df, 10),
                        'sma_20': lambda df: Ind.sma(df, 20),
                        'ema_20': lambda df: Ind.ema(df, 20),
                        'sma_bolling': lambda df: Ind.boll(df, 20, 2.0),
                        'ema_bolling': lambda df: Ind.boll(df, 20, 2.0, 'ema'),
                        'day_extremum': lambda df: Ind.day_extremum(df),
                        'zigzag': lambda df: Ind.zigzag(df, 0.2, mode='ratio'),
                    },
                ),
                # 合成30分钟的数据及指标的配置
                'min30': set_BackTestingDataSetting(
                    data_source=data_source,
                    new_bar_interval=Sec.MIN30.value,  # 新bar 周期
                    indicators={  # 需要预处理的数据指标
                        'sma_5': lambda df: Ind.sma(df, 5),
                        'ema_5': lambda df: Ind.ema(df, 5),
                        'sma_10': lambda df: Ind.sma(df, 10),
                        'ema_10': lambda df: Ind.ema(df, 10),
                        'sma_20': lambda df: Ind.sma(df, 20),
                        'ema_20': lambda df: Ind.ema(df, 20),
                        'sma_60': lambda df: Ind.sma(df, 60),
                        'ema_60': lambda df: Ind.ema(df, 60),
                    },
                )
            },  # self.data_sets

            # 回测需要的超参数 list
            hyper_parameter=[{"ma_type": ma_type,
                              'bolling_len': bolling_len,
                              "bl_for_long": bl_for_long,
                              # "bl_for_short": bl_for_short,
                              'decide_open_type': decide_open_type,  # 开仓判断时用什么的价格
                              "stop_loss_type": stop_loss_type,
                              'take_profit_type': take_profit_type,
                              }  # 止损方式
                             for ma_type in ['ema', 'sma']
                             for bolling_len in [20]
                             for bl_for_long in ["up", "mid", "down"]
                             # for bl_for_short in ["up", "mid", "down"]
                             for decide_open_type in ['extreme', 'Close']
                             for stop_loss_type in ['boll_II', 'boll_I', 'bar'][:]
                             for take_profit_type in ['boll_II', ]
                             ][:],

            # 交易参数 开平仓等
            trade_params=TradeParams(
                symbol=data_source.symbol[:2],
                init_money=10000,  # 初始金额 USD
                max_lots=10,
                once_lots=1,
                open_max=OpenMax.LONG_1_OR_SHORT_1,  # 开仓限制
                # fee_rate= 0.1,  # 交易手续费 %（可选，优先于fee_amount）
                fee_amount=5.0,  # 交易手续费 固定量（可选）
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

        # 切片数据  curr当前bar  prior前bar 。。。
        prior_min1, curr_min1 = used_data['min1'].iloc[0], used_data['min1'].iloc[1]
        prior_min5, curr_min5 = used_data['min5'].iloc[0], used_data['min5'].iloc[1]
        prior_min30, curr_min30 = used_data['min30'].iloc[0], used_data['min30'].iloc[1]

        # 回测超参数 阈值等 -----------------------------
        ma_type = params.get('ma_type', )
        bolling_len = params.get('bolling_len', [])
        bl_for_long = params.get('bl_for_long', [])  # 单个或者list
        if isinstance(bl_for_long, str):
            bl_for_long = [bl_for_long, ]
        # bl_for_short = params.get('bl_for_short', )  # 单个或者list
        # if isinstance(bl_for_short, str):
        #     bl_for_short = [bl_for_short,]
        bl_for_short = bl_for_long[::-1]  # long-short逆序
        decide_open_type = params.get('decide_open_type', )  # 开仓判断时用什么的价格
        stop_loss_type = params.get('stop_loss_type', )  # 止损方式
        take_profit_type = params.get('take_profit_type', )  # 止盈方式

        # 交易参数
        trade_p: TradeParams = self.trade_params  # 交易参数

        # 参数本身取值范围
        days = {'prior': 0, 'curr': 1}
        k_periods = {'I': 1, 'II': 5, 'III': 30}  # k线周期
        ma_lens = {'I': 5, 'II': 10, 'III': 20, 'IV': 60}

        # 具体数据
        try:
            min30_ma = {}  # 当前bar 30分钟 MA的数值
            for mat, mal in ma_lens.items():
                min30_ma[mat] = curr_min30[f'{ma_type}_{mal}']

            min5_ma = {}  # 当前bar 5分钟 MA的数值
            for mat in ['I', 'II', 'III']:
                min5_ma[mat] = curr_min5[f'{ma_type}_{ma_lens[mat]}']

            bolling = {}
            for day, dv in days.items():
                if day not in bolling:
                    bolling[day] = {}
                for p_ in ['I', 'II']:  # k_periods
                    if p_ not in bolling[day]:
                        bolling[day][p_] = {}
                    for t in ["up", "mid", "down"]:
                        bolling[day][p_][t] = used_data[f'min{k_periods[p_]}'].iloc[dv][
                            f'boll_{t}_{ma_type}_{bolling_len}']
        except Exception as e:
            self.log(f'data err: {e}')

        # 当前处理的bar的index
        curr_index = used_data['min1'].index[-1]

        # self.log(f'curr_index = {curr_index}  {used_data["min1"]["datetime"]}')

        def _process(trade: SimuTrade) -> SimuTrade:
            """具体逻辑"""

            direction = Direction.NONE  # 交易方向

            def open_long() -> bool:
                """开多判断"""
                # 30min
                # 短期均线MA(5,10,20日均线）>60日
                if not (min30_ma['I'] > min30_ma['IV']
                        and min30_ma['II'] > min30_ma['IV']
                        and min30_ma['III'] > min30_ma['IV']):
                    # self.log(f'open_long 30min "短期均线MA(5,10,20日均线）>60日" 不满足')
                    return False

                # 5min
                # 5日均线向上突破10日均线突破20日均线（10可以小于20）
                # or 10日均线向上突破20如均线判定为多头
                if not ((min5_ma['I'] > min5_ma['II'] and min5_ma['I'] > min5_ma['III'])
                        or (min5_ma['II'] > min5_ma['III'])):
                    # self.log(f'open_long 5min "5日均线向上突破10日均线突破20日均线（10可以小于20）or 10日均线向上突破20如均线判定为多头" 不满足')
                    return False

                # 1min
                # 是阳线
                if curr_min1.Last <= curr_min1.Open:
                    # self.log(f'open_long 1min "是阳线" 不满足')
                    return False
                # open在轨道下方，且布林带实时价格向上破 中轨/下轨/上轨 （或组合）
                used_price_ = (curr_min1.Last if decide_open_type == 'Close'
                               else curr_min1.High)
                for blt in bl_for_long:
                    if (prior_min1.Last < bolling['prior']['I'][blt] and
                            used_price_ > bolling['curr']['I'][blt] > curr_min1.Open):
                        # self.log(f'open_long 1min  **** 满足')
                        return True  # 穿过一条即可

                # self.log(f'open_long 1min "open在轨道下方，且布林带实时价格向上破 中轨/下轨/上轨 " 不满足')
                return False

            def open_short() -> bool:
                """开空判断"""
                # 30min
                # 短期均线MA(5,10,20日均线）<60日
                if not (min30_ma['I'] < min30_ma['IV']
                        and min30_ma['II'] < min30_ma['IV']
                        and min30_ma['III'] < min30_ma['IV']):
                    # self.log(f'open_short 30min "短期均线MA(5,10,20日均线）<60日" 不满足')
                    return False

                # 5min
                # 5日均线向下突破10日均线突破20日均线（10可以小于20）
                # or 10日均线向下突破20如均线判定为空头
                if not ((min5_ma['I'] < min5_ma['II'] and min5_ma['I'] < min5_ma['III'])
                        or (min5_ma['II'] < min5_ma['III'])):
                    # self.log(f'open_short 5min 5日均线向下突破10日均线突破20日均线" 不满足')
                    return False

                # 1min
                # 是阴线
                if curr_min1.Last >= curr_min1.Open:
                    # self.log(f'open_short 1min 阴线" 不满足')
                    return False
                # open在轨道上方，且布林带实时价格向下破 中轨/下轨/上轨 （或组合）
                used_price_ = (curr_min1.Last if decide_open_type == 'Close'
                               else curr_min1.Low)
                for blt in bl_for_short:
                    if (prior_min1.Last > bolling['prior']['I'][blt] and
                            used_price_ < bolling['curr']['I'][blt] < curr_min1.Open):
                        # self.log(f'open_short 1min **** 满足')
                        return True  # 穿过一条即可

                # self.log(f'open_short 1min open在轨道上方，且布林带实时价格向下破 中轨/下轨/上轨 不满足')
                return False

            def close_long() -> bool:
                """平多判断"""
                # 止盈  5分钟布林带上轨外
                if curr_min1.Last > bolling['curr']['II']['up']:
                    return True

                p = {'bar': trade.stop_loss,  # bar low
                     'boll_I': bolling['curr']['I']['down'],
                     'boll_II': bolling['curr']['II']['down'],
                     }
                if curr_min1.Last < p[stop_loss_type]:
                    return True
                else:
                    return False

            def close_short() -> bool:
                """平空判断"""
                # 止盈  5分钟布林带下轨外
                if curr_min1.Last < bolling['curr']['II']['down']:
                    # self.log(f"close_short  不满足 {curr_min1.Last} < {bolling['curr']['II']['down']}")
                    return True

                p = {'bar': trade.stop_loss,  # bar high
                     'boll_I': bolling['curr']['I']['up'],
                     'boll_II': bolling['curr']['II']['up'],
                     }
                if curr_min1.Last > p[stop_loss_type]:
                    return True
                else:
                    # self.log(f"close_short  不满足 curr_min1.Last{curr_min1.Last} > {p[stop_loss_type]}")
                    return False

            # 逻辑开始 ==================
            if trade.waiting_open:  # 开仓进行判断
                if open_long():
                    direction = Direction.SHORT  # 开多
                elif open_short():
                    direction = Direction.LONG  # 开空

                # 满足开仓 用SimTrade.set_open
                if direction != Direction.NONE:
                    trade.set_open(direction,
                                   curr_index,  # 当前bar
                                   curr_min1['Last'],  # 当前价
                                   amount=trade_p.once_lots,  # 开仓量
                                   datetime_=curr_min1['datetime'],  # 开仓时间
                                   stop_loss=(curr_min1['Low'] if direction == Direction.LONG
                                              else curr_min1['High']),
                    )

                return trade

            else:  # 平仓进行判断
                # 更新最大浮盈
                trade.record_extreme(high=curr_min1.High, low=curr_min1.Low)

                # 多头平仓
                if (trade.direction == Direction.LONG
                        and close_short()):
                    direction = Direction.LONG

                # 空头平仓
                if (trade.direction == Direction.SHORT
                        and close_long()):
                    direction = Direction.SHORT

                # 满足平仓 用SimTrade.set_close 数量==开仓数量
                if direction in [Direction.LONG, Direction.SHORT]:
                    trade.set_close(curr_index,  # 当前bar
                                    curr_min1['Last'],  # 当前价
                                    curr_min1['datetime'],  # 平仓时间
                                    )
                # 返回最后结果
                return trade

        # 遍历各交易(未开仓/已开仓未平仓）
        result = []
        for each_trade in working_trades:
            try:
                each_trade = _process(each_trade)
            except Exception as e:
                self.log(f'work err: {e}', logging.ERROR)
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

if __name__ == '__main__':
    p = path.abspath(__file__)
    print(p, __file__)
    import importlib

    # c = importlib.import_module(p)
    print(path.split(path.realpath(__file__)))
