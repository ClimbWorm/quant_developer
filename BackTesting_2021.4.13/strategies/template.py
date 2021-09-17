#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略模板

# Author: Tayii
# Data : 2020/12/02
# ----------------------------------------------------
import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum, unique

from backtester.template import BackTestingDataSetting
from datahub.fmt import SimuTrade
from my_log import ilog
import logging
from typing import Any, Callable, Dict, List, Optional
from abc import ABC, abstractmethod

from constant import Direction, Offset, Interval, OpenMax, TickPara, SymbolTickPara
from utility import catch_except


# def select_hyper_parameter(plan_name: str) -> List[dict]:
#     """选中的部分超参数 一般用作二次回测"""
#     columns = list(hyper_parameter[0].keys())
#     total_csv_filepath = path.join(BACK_TESTING_RESULT_DIR,
#                                    f'{plan_name}', 'total_result.csv')
#     df = pd.read_csv(total_csv_filepath)
#     df = df[df['total_net_profit.all'] > 0]
#     df = df[columns]
#     select_params: List[dict] = df.to_dict('records')
#     # 对原始数据进行处理
#     for s in select_params:
#         s['min1_ma_periods'] = eval(s['min1_ma_periods'])  # 字符串变数组
#         s['min1_bolling_dev'] = (s['min1_bolling_dev'], -s['min1_bolling_dev'])
#     return select_params


@unique
class OpenCloseTypes(Enum):
    """ 平仓类型 """
    REVERSE_BREAK_MAIN_BL = 'reverse_break_self_bolling'  # 反向破主布林带
    REVERSE_BREAK_SMALLER_BL = 'reverse_break_smaller_bolling'  # 反向破更小周期
    CROSS_DO_BL = 'cross_do_bolling'  # 回穿操作布林带
    CROSS_BL_ACCORDING_TO_BASE = 'cross_bolling_according_to_base'  # 基于主方向的 强回穿主布林带
    CROSS_SMALLER_BL = 'cross_smaller_bolling'  # 小周期的强回穿布林带
    CROSS_SELF_BL_PLUS = 'cross_self_bl_plus'  # 突破布林带后 多策略开平仓
    CROSS_AND_BREAK_BL = 'cross_and_break_bl'  # 突破和回踩布林带组合 多策略开平仓
    CROSS_SUPPORT_MID = 'cross_support_mid'  # 支撑均线的强回穿
    CROSS_ALL_SUPPORT_MID = 'cross_all_support_mid'  # 所有支撑均线的强回穿
    IN_BL_OUTSIDE_PLUS = 'in_bolling_outside_plus'  # 在布林带外 多策略，（一般开仓）
    SPECIAL_FOR_144 = 'special_for_144'
    TOUCH_BL_OUTSIDE_PLUS = 'touch_bolling_outside_plus'  # 触摸到布林带外后 多策略
    TOUCH_BL_AND_CROSS_MID = 'touch_bolling_and_cross_mid'  # 触摸到布林带外后 破中线
    DEV_RANGE = 'dev_range'  # 布林带大区域
    DEV_RANGE_TREND = 'dev_range_trend'  # 布林带大区域+顺势
    DYNAMIC_DEV = 'dynamic_dev'  # 动态布林带区域
    TRIGGER = 'trigger'  # 动态止盈止损
    FIXED_STOP_LOSS = 'fixed_stop_loss'  # 固定止损
    JUST_MA_TREND = 'just_ma_trend'  # 仅ma趋势
    MA_TREND_OUTSIDE_BL = 'ma_trend_outside_bl'  # 布林带大区域+ma趋势
    STOP_LOSS_OVER_TIME = 'stop_loss_over_time'  # 超时止损
    DAY_END = 'day_end'  # 日收盘
    OPPOSITE_DIRECTION = 'opposite_direction',  # 反向信号
    AFTER_T = 'after_t'  # 过了一定时间后
    # 移动止盈
    # 固定止损


@dataclass
class OpenCloseParams:
    """ 保存 开平仓相关参数 """
    direction: Direction = Direction.NONE  # 是否要平仓/平仓
    type: Any = None  # 平仓/平仓触发类型
    para: Any = None  # 平仓/平仓 参数
    price: float = None  # 平仓/平仓价
    bar: int = None  #

    @property
    def is_valid(self)->bool:
        return self.direction != Direction.NONE

@dataclass
class TradeParams:
    """ 交易相关参数 """
    symbol: str
    init_money: float  # 初始金额 USD
    max_lots: int
    once_lots: int  # 单次开仓手数
    open_max: OpenMax  # 开仓限制
    fee_rate: Optional[float] = None  # 交易手续费 %（可选，优先于fee_amount）
    fee_amount: Optional[float] = None  # 交易手续费 固定量（可选）
    slippage: int = 0,  # 交易滑点  tick数
    usd_per_tick: float = field(init=False)  # 每个tick价值n美金
    tick_size: float = field(init=False)  # tick size

    @catch_except()
    def __post_init__(self):
        if self.fee_rate is None and self.fee_amount is None:
            return  # 不能同时None
        tp: TickPara = SymbolTickPara[self.symbol[:2]]
        self.usd_per_tick = tp.usd_per_tick
        self.tick_size = tp.tick_size


class StrategyTemplate():
    """策略类 模板"""

    def __init__(self,
                 name: str,
                 need_bars_once: int,  # 一次切片回测需要最近几个bar数据
                 data_sets: Dict[str, BackTestingDataSetting],  # 回测用 数据配置文件
                 hyper_parameter: List[dict],  # 超参数组合
                 trade_params: TradeParams,  # 交易参数 开平仓等
                 result_paras: Dict,  # 盈利计算模式 参数
                 save_data_result: bool = True,  # 保存回测数据结果
                 show_data_result: bool = False,  # 显示回测数据结果
                 save_trade_result: bool = True,  # 保存回测交易结果
                 import_path: str = '',  # 导入用 包路径
                 author: str = "Tayii",
                 ):
        self.name: str = name
        self.need_bars_once: int = need_bars_once  # 一次切片回测需要最近几个bar数据
        self.data_sets: Dict[str, BackTestingDataSetting] = data_sets  # 回测用 数据配置文件
        self.hyper_parameter: List[dict] = hyper_parameter  # 超参数组合
        self.trade_params: TradeParams = trade_params  # 交易参数 开平仓等
        self.result_paras: Dict = result_paras  # 盈利计算模式 参数

        self.save_data_result: bool = save_data_result  # 保存回测数据结果
        self.show_data_result: bool = show_data_result  # 显示回测数据结果
        self.save_trade_result: bool = save_trade_result  # 保存回测交易结果

        self.import_path: str = import_path  # 导入用 包路径
        self.author: str = author

    @staticmethod
    def close_trade(trade: SimuTrade,  # 平仓的工单
                    close_bar: int,  # 当前bar
                    close_type: str,  # 平仓类型
                    close_price: float,  #
                    datetime_: datetime,  # 平仓时间
                    close_para: Any = None,  # 平仓参数
                    ):
        """平仓原有的订单"""
        trade.set_close(close_bar=close_bar,  # 当前bar
                        close_type=close_type,
                        close_price=close_price,  #
                        datetime_=datetime_,  # 平仓时间
                        close_para=close_para,
                        )

    def __str__(self):
        return self.name

    __repr__ = __str__

    def log(self, msg, level=logging.DEBUG, exc_info=False) -> None:
        ilog.console(f'{self.name}: {msg}', level=level, exc_info=exc_info)

# class StrategyTemplate(ABC):
#     """
#     Template for strategies.
#     """
#
#     author = ""
#     parameters = []
#     variables = []
#
#     def __init__(self, engine: Any, strategy_name: str,
#                  e_symbol: str, setting: dict, ):
#         """"""
#         self.engine = engine
#         self.strategy_name = strategy_name
#         self.e_symbol = e_symbol
#
#         self.inited = False
#         self.trading = False
#         self.pos = 0
#
#         # Copy a new variables list here to avoid duplicate insert when multiple
#         # strategy instances are created with the same strategy class.
#         self.variables = copy(self.variables)
#         self.variables.insert(0, "inited")
#         self.variables.insert(1, "trading")
#         self.variables.insert(2, "pos")
#
#         self.update_setting(setting)  # 更新参数
#
#     def update_setting(self, setting: dict):
#         """
#         Update strategy parameter with value in setting dict.
#         """
#         for name in self.parameters:
#             if name in setting:
#                 setattr(self, name, setting[name])
#
#     @classmethod
#     def get_class_parameters(cls):
#         """
#         Get default parameters dict of strategy class.
#         """
#         class_parameters = {}
#         for name in cls.parameters:
#             class_parameters[name] = getattr(cls, name)
#         return class_parameters
#
#     def get_parameters(self):
#         """
#         Get strategy parameters dict.
#         """
#         strategy_parameters = {}
#         for name in self.parameters:
#             strategy_parameters[name] = getattr(self, name)
#         return strategy_parameters
#
#     def get_variables(self):
#         """
#         Get strategy variables dict.
#         """
#         strategy_variables = {}
#         for name in self.variables:
#             strategy_variables[name] = getattr(self, name)
#         return strategy_variables
#
#     def get_data(self):
#         """
#         Get strategy data.
#         """
#         strategy_data = {
#             "strategy_name": self.strategy_name,
#             "e_symbol": self.e_symbol,
#             "class_name": self.__class__.__name__,
#             "author": self.author,
#             "parameters": self.get_parameters(),
#             "variables": self.get_variables(),
#         }
#         return strategy_data
#
#     @abstractmethod
#     def on_init(self):
#         """
#         Callback when strategy is inited.
#         """
#         pass
#
#     @abstractmethod
#     def on_start(self):
#         """
#         Callback when strategy is started.
#         """
#         pass
#
#     @abstractmethod
#     def on_stop(self):
#         """
#         Callback when strategy is stopped.
#         """
#         pass
#
#     @abstractmethod
#     def on_tick(self, tick: TickData):
#         """
#         Callback of new tick data update.
#         """
#         pass
#
#     @abstractmethod
#     def on_bar(self, bar: BarData):
#         """
#         Callback of new bar data update.
#         """
#         pass
#
#     @abstractmethod
#     def on_trade(self, trade: TradeData):
#         """
#         Callback of new trade data update.
#         """
#         pass
#
#     @abstractmethod
#     def on_order(self, order: OrderData):
#         """
#         Callback of new order data update.
#         """
#         pass
#
#     @abstractmethod
#     def on_stop_order(self, stop_order: StopOrder):
#         """
#         Callback of stop order update.
#         """
#         pass
#
#     def buy(self, price: float, volume: float, stop: bool = False, lock: bool = False):
#         """
#         Send buy order to open a long position.
#         """
#         return self.send_order(Direction.LONG, Offset.OPEN, price, volume, stop, lock)
#
#     def sell(self, price: float, volume: float, stop: bool = False, lock: bool = False):
#         """
#         Send sell order to close a long position.
#         """
#         return self.send_order(Direction.SHORT, Offset.CLOSE, price, volume, stop, lock)
#
#     def short(self, price: float, volume: float, stop: bool = False, lock: bool = False):
#         """
#         Send short order to open as short position.
#         """
#         return self.send_order(Direction.SHORT, Offset.OPEN, price, volume, stop, lock)
#
#     def cover(self, price: float, volume: float, stop: bool = False, lock: bool = False):
#         """
#         Send cover order to close a short position.
#         """
#         return self.send_order(Direction.LONG, Offset.CLOSE, price, volume, stop, lock)
#
#     def send_order(
#             self,
#             direction: Direction,
#             offset: Offset,
#             price: float,
#             volume: float,
#             stop: bool = False,
#             lock: bool = False
#     ):
#         """
#         Send a new order.
#         """
#         if self.trading:
#             vt_orderids = self.engine.send_order(
#                 self, direction, offset, price, volume, stop, lock
#             )
#             return vt_orderids
#         else:
#             return []
#
#     def cancel_order(self, vt_orderid: str):
#         """
#         Cancel an existing order.
#         """
#         if self.trading:
#             self.engine.cancel_order(self, vt_orderid)
#
#     def cancel_all(self):
#         """
#         Cancel all orders sent by strategy.
#         """
#         if self.trading:
#             self.engine.cancel_all(self)
#
#     def write_log(self, msg: str):
#         """
#         Write a log message.
#         """
#         self.engine.write_log(msg, self)
#
#     def get_engine_type(self):
#         """
#         Return whether the cta_engine is backtester or live trading.
#         """
#         return self.engine.get_engine_type()
#
#     def get_pricetick(self):
#         """
#         Return pricetick data of trading contract.
#         """
#         return self.engine.get_pricetick(self)
#
#     def load_bar(
#             self,
#             days: int,
#             interval: Interval = Interval.MINUTE,
#             callback: Callable = None,
#             use_database: bool = False
#     ):
#         """
#         Load historical bar data for initializing strategy.
#         """
#         if not callback:
#             callback = self.on_bar
#
#         self.engine.load_bar(
#             self.e_symbol,
#             days,
#             interval,
#             callback,
#             use_database
#         )
#
#     def load_tick(self, days: int):
#         """
#         Load historical tick data for initializing strategy.
#         """
#         self.engine.load_tick(self.e_symbol, days, self.on_tick)
#
#     def put_event(self):
#         """
#         Put an strategy data event for ui update.
#         """
#         if self.inited:
#             self.engine.put_strategy_event(self)
#
#     def send_email(self, msg):
#         """
#         Send email to default receiver.
#         """
#         if self.inited:
#             self.engine.send_email(msg, self)
#
#     def sync_data(self):
#         """
#         Sync strategy variables value into disk storage.
#         """
#         if self.trading:
#             self.engine.sync_strategy_data(self)
