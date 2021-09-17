#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# ----------------------------------------------------
# 回测引擎
# Author: Tayii
# Data : 2020/12/01
# ----------------------------------------------------
from datetime import date, datetime, timedelta

from constant import EngineType, BacktestingMode, Interval, Exchange

#
# class BackTestingEngine(BaseEngine):
#     """"""
#
#     engine_type = EngineType.BACKTESTING
#     source_name = "BACK_TESTING"
#
#     def __init__(self, event_engine: EventEngine):
#         """"""
#         BaseEngine.__init__(self, event_engine, "back_testing")
#         self.symbol = ""
#         self.exchange = None
#         self.start = None
#         self.end = None
#         self.rate = 0
#         self.slippage = 0
#         self.size = 1
#         self.pricetick = 0
#         self.capital = 1_000_000
#         self.mode = BacktestingMode.BAR
#         self.inverse = False
#
#         self.strategy_class = None
#         self.strategy = None
#         self.tick: TickData
#         self.bar: BarData
#         self.datetime = None
#
#         self.interval = None
#         self.days = 0
#         self.callback = None
#         self.history_data = []
#
#         self.stop_order_count = 0
#         self.stop_orders = {}
#         self.active_stop_orders = {}
#
#         self.limit_order_count = 0
#         self.limit_orders = {}
#         self.active_limit_orders = {}
#
#         self.trade_count = 0
#         self.trades = {}
#
#         self.logs = []
#
#         self.daily_results = {}
#         self.daily_df = None
#
#     def clear_data(self):
#         """
#         Clear all data of last backtester.
#         """
#         self.strategy = None
#         self.tick = None
#         self.bar = None
#         self.datetime = None
#
#         self.stop_order_count = 0
#         self.stop_orders.clear()
#         self.active_stop_orders.clear()
#
#         self.limit_order_count = 0
#         self.limit_orders.clear()
#         self.active_limit_orders.clear()
#
#         self.trade_count = 0
#         self.trades.clear()
#
#         self.logs.clear()
#         self.daily_results.clear()
#
#     def set_parameters(self, e_symbol: str, interval: Interval,
#                        start: datetime, rate: float, slippage: float,
#                        size: float, price_tick: float, capital: int = 0,
#                        end: datetime = None, mode: BacktestingMode = BacktestingMode.BAR,
#                        inverse: bool = False):
#         """"""
#         self.mode = mode
#         self.e_symbol = e_symbol
#         self.interval = Interval(interval)
#         self.rate = rate
#         self.slippage = slippage
#         self.size = size
#         self.price_tick = price_tick
#         self.start = start
#
#         self.symbol, exchange_str = self.e_symbol.split(".")
#         self.exchange = Exchange(exchange_str)
#
#         self.capital = capital
#         self.end = end
#         self.mode = mode
#         self.inverse = inverse
#
#     def add_strategy(self, strategy_class: type, setting: dict):
#         """"""
#         self.strategy_class = strategy_class
#         self.strategy = strategy_class(
#             self, strategy_class.__name__, self.vt_symbol, setting
#         )
#
#     # def cancel_order(self, strategy: CtaTemplate, vt_orderid: str):
#     #     """
#     #     Cancel order by vt_orderid.
#     #     """
#     #     if vt_orderid.startswith(STOPORDER_PREFIX):
#     #         self.cancel_stop_order(strategy, vt_orderid)
#     #     else:
#     #         self.cancel_limit_order(strategy, vt_orderid)
#     #
#     # def cancel_stop_order(self, strategy: CtaTemplate, vt_orderid: str):
#     #     """"""
#     #     if vt_orderid not in self.active_stop_orders:
#     #         return
#     #     stop_order = self.active_stop_orders.pop(vt_orderid)
#     #
#     #     stop_order.status = StopOrderStatus.CANCELLED
#     #     self.strategy.on_stop_order(stop_order)
#     #
#     # def cancel_limit_order(self, strategy: CtaTemplate, vt_orderid: str):
#     #     """"""
#     #     if vt_orderid not in self.active_limit_orders:
#     #         return
#     #     order = self.active_limit_orders.pop(vt_orderid)
#     #
#     #     order.status = Status.CANCELLED
#     #     self.strategy.on_order(order)
#     #
#     # def cancel_all(self, strategy: CtaTemplate):
#     #     """
#     #     Cancel all orders, both limit and stop.
#     #     """
#     #     vt_orderids = list(self.active_limit_orders.keys())
#     #     for vt_orderid in vt_orderids:
#     #         self.cancel_limit_order(strategy, vt_orderid)
#     #
#     #     stop_orderids = list(self.active_stop_orders.keys())
#     #     for vt_orderid in stop_orderids:
#     #         self.cancel_stop_order(strategy, vt_orderid)
#
#     def write_log(self, msg: str, ):  # strategy: CtaTemplate = None
#         """
#         Write log message.
#         """
#         msg = f"{self.datetime}\t{msg}"
#         self.logs.append(msg)
#
#     # def send_email(self, msg: str, strategy: CtaTemplate = None):
#     #     """
#     #     Send email to default receiver.
#     #     """
#     #     pass
#
#     # def sync_strategy_data(self, strategy: CtaTemplate):
#     #     """
#     #     Sync strategy data into json file.
#     #     """
#     #     pass
#
#     def get_engine_type(self):
#         """
#         Return engine type.
#         """
#         return self.engine_type
#
#     def get_price_tick(self, ):  # strategy: CtaTemplate
#         """
#         Return contract pricetick data.
#         """
#         return self.price_tick
#
#     # def put_strategy_event(self, strategy: CtaTemplate):
#     #     """
#     #     Put an event to update strategy status.
#     #     """
#     #     pass
#
#     def output(self, msg):
#         """
#         Output message of backtester engine.
#         """
#         print(f"{datetime.now()}\t{msg}")
#
#     def get_all_trades(self):
#         """
#         Return all trade data of current backtester result.
#         """
#         return list(self.trades.values())
#
#     def get_all_orders(self):
#         """
#         Return all limit order data of current backtester result.
#         """
#         return list(self.limit_orders.values())
#
#     def get_all_daily_results(self):
#         """
#         Return all daily result data.
#         """
#         return list(self.daily_results.values())
