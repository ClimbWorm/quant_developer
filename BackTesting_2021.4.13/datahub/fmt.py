#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# Basic data structure used for general trading function
#
# Author: Tayii
# Data : 2021/1/18
# ----------------------------------------------------
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Any

import pandas as pd

from constant import (Direction, Exchange, Interval, Offset,
                      Status, Product, OptionType, OrderType,
                      StopOrderStatus)
from datahub.status import BlDevStatus, PriceStatus
from my_log import ilog

ACTIVE_STATUSES = {Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED}


class DataBase:
    """
    Any data object needs a name as source
    and should inherit base data.
    """

    def __init__(self, name):
        self.name: str = name

    def log(self, msg, level=logging.DEBUG, exc_info=False):
        """ 此类专属打印 """
        msg = f"[{self.name}]: {msg}"
        return ilog.console(msg, level, exc_info=exc_info)

    def write_log(self, msg, level=logging.ERROR):
        """打印 写入文件"""
        msg = f"[{self.name}]: {msg}"
        ilog.to_file(msg, level=level, )


@dataclass
class SimuTrade:
    """
    单笔完整交易
    回测用的 模拟交易格式
    """
    # symbol: str
    usd_per_tick: float  # 每个tick价值n美金  # todo 最好放symbol里
    tick_size: float  # tick size
    fee_rate: float = 0.0  # 交易手续费 %（可选，优先于fee_amount）
    fee_amount: float = 0.0  # 交易手续费 固定量（可选）
    slippage: int = 0  # 滑点  tick数
    open_direction: Direction = Direction.NONE  # 开仓交易方向
    waiting_condition: Any = None  # 开仓等待的条件 可选
    open_bar: int = -1  # 开仓bar index
    open_price: float = 0.0  # 开仓价
    open_amount: float = 0.0  # 开仓数据（平仓数量==开仓数量）即必须平完
    open_datetime: datetime = None  # 开仓时间
    open_type: str = None  # 开仓类型
    open_para: Any = None  # 开仓参数
    dev_status: BlDevStatus = None  # dev状态
    price_status: PriceStatus = None  # price状态
    close_datetime: datetime = None  # 平仓时间
    close_type: str = None  # 平仓类型
    close_para: Any = None  # 平仓参数
    close_bar: int = -1  # 平仓bar index
    close_price: float = 0.0  # 平仓价
    # close_amount: float = 0.0  # 与open_amount一样
    # 开仓后处理
    earnings: float = 0.0  # 收益
    max_floating_profit: float = -999999  # 最大浮盈
    max_floating_loss: float = 999999  # 最大浮亏
    duration: float = 0.0  # 持续时间 秒
    highest_v: float = 0  # 中间记录的值 最大点
    highest_index: int = -1  # ban index
    highest_at: int = 0  # 在过了多少个bar后出现
    lowest_v: float = 9999999  # 中间记录的值 最小点
    lowest_index: int = -1  # ban index
    lowest_at: int = 0  # 在过了多少个bar后出现

    def __str__(self):
        return f'trade: open: {self.open_direction} {self.open_price} ' \
               f'{self.open_amount}  - finished={self.finished}'

    __repr__ = __str__

    def __eq__(self, other):
        """特别针对 新开的订单(同一个bar不能同方向开2单) """
        return (True if self.open_direction == other.open_direction
                        and self.open_bar == other.open_bar
                        and self.waiting_condition == other.waiting_condition
                else False)

    def __hash__(self):
        return hash(f'{self.open_direction}{self.open_bar}')

    def record_extreme(self, high, low, bar_index):
        """记录极值点"""
        if high > self.highest_v:
            self.highest_v = high
            self.highest_index = bar_index
        if low < self.lowest_v:
            self.lowest_v = low
            self.lowest_index = bar_index

    @property
    def opening(self) -> bool:
        """订单是否存活(=已开仓未平仓)"""
        return self.open_datetime and (self.close_datetime is None)

    @property
    def waiting_open(self) -> bool:
        """此笔交易 是否还未开始（等待开仓）"""
        return self.open_datetime is None

    @property
    def is_black_trade(self) -> bool:
        """此笔交易 是否是空白单（即还未开始，且不是延续单）"""
        return self.waiting_open and self.waiting_condition is None

    @property
    def waiting_close(self) -> bool:
        """此笔交易 是否等待平仓"""
        return self.opening

    @property
    def finished(self) -> bool:
        """此笔交易 是否已结束"""
        return self.close_datetime is not None

    @property
    def direction(self):
        """
        持仓方向 == 开仓方向
        PS: 新加时间判断，因为中继单有时候会预设方向
        """
        return self.open_direction if self.open_datetime else Direction.NONE

    def set_open(self,
                 direction_,  # 开仓方向
                 bar: int,  # 开仓的bar
                 price: float,  # 开仓价
                 amount: float,  # 开仓量
                 datetime_: datetime,  # 开仓时间
                 open_type: str = ' ',  # 开仓类型
                 open_para: Any = None,  # 开仓参数
                 **kwargs,
                 ):
        """设置开仓参数"""
        self.open_direction = direction_
        self.open_bar = bar
        self.open_price = price
        self.open_amount = amount
        self.open_datetime = datetime_
        self.open_type = open_type
        self.open_para = open_para
        self.dev_status = kwargs.get('dev_status')
        self.price_status = kwargs.get('price_status')


    def set_close(self,
                  close_bar: int,  # 平仓bar index
                  close_type: str,  # 平仓类型
                  close_price: float,  # 平仓价
                  datetime_: datetime,  # 平仓时间
                  close_para: Any = None,  # 开仓参数
                  ) -> None:
        """平仓后 对参数进行更新"""
        self.close_bar = close_bar
        self.close_price = close_price
        self.close_datetime = datetime_
        self.close_type = close_type
        self.close_para = close_para
        # 简单统计下这笔交易
        self.duration = (self.close_datetime - self.open_datetime).total_seconds()
        # 计算单手盈亏
        profit_1lot = (self.close_price - self.open_price) / self.tick_size * self.usd_per_tick
        if self.open_direction == Direction.SHORT:
            profit_1lot = -profit_1lot
        # 减去手续费
        profit_1lot = self.__minus_fee(profit_1lot)
        # 单次交易总的利润
        self.earnings = profit_1lot * self.open_amount

        # 计算/更新最大浮盈浮亏
        p_lot = {}
        for t in ['highest_v', 'lowest_v']:
            price = eval(f'self.{t}')
            p_lot[t] = (price - self.open_price) / self.tick_size * self.usd_per_tick
            if self.open_direction == Direction.SHORT:
                p_lot[t] *= -1
            p_lot[t] = self.__minus_fee(p_lot[t])  # 单手最大浮盈浮亏
            p_lot[t] *= self.open_amount  # 总的

        self.max_floating_profit = p_lot['highest_v'] if self.direction == Direction.LONG else p_lot['lowest_v']
        self.max_floating_loss = p_lot['lowest_v'] if self.direction == Direction.LONG else p_lot['highest_v']

        self.highest_at = self.highest_index - self.open_bar
        self.lowest_at = self.lowest_index - self.open_bar

    def __minus_fee(self, profit_1lot: float) -> float:
        """减去手续费 滑点等费用（剩余的利润） 单手"""
        if self.fee_rate:
            profit_1lot -= (self.open_price + self.close_price) / 2 * self.fee_rate / 100
        elif self.fee_amount:
            profit_1lot -= self.fee_amount
        else:
            return 0
        # 减去滑点
        profit_1lot -= self.slippage * self.usd_per_tick
        return profit_1lot

    def result_as_series(self) -> pd.Series:
        """以series返回结果"""
        return pd.Series(asdict(self))


# #
# # @dataclass
# # class TickData(DataBase):
# #     """
# #     Tick data contains information about:
# #         * last trade in market
# #         * orderbook snapshot
# #         * intraday market statistics.
# #     """
# #     DataBase.__init__()
# #     symbol: str
# #     exchange: Exchange
# #     datetime: datetime
# #
# #     name: str = ""
# #     volume: float = 0
# #     open_interest: float = 0
# #     last_price: float = 0
# #     last_volume: float = 0
# #     limit_up: float = 0
# #     limit_down: float = 0
# #
# #     open_price: float = 0
# #     high_price: float = 0
# #     low_price: float = 0
# #     pre_close: float = 0
# #
# #     bid_price_1: float = 0
# #     bid_price_2: float = 0
# #     bid_price_3: float = 0
# #     bid_price_4: float = 0
# #     bid_price_5: float = 0
# #
# #     ask_price_1: float = 0
# #     ask_price_2: float = 0
# #     ask_price_3: float = 0
# #     ask_price_4: float = 0
# #     ask_price_5: float = 0
# #
# #     bid_volume_1: float = 0
# #     bid_volume_2: float = 0
# #     bid_volume_3: float = 0
# #     bid_volume_4: float = 0
# #     bid_volume_5: float = 0
# #
# #     ask_volume_1: float = 0
# #     ask_volume_2: float = 0
# #     ask_volume_3: float = 0
# #     ask_volume_4: float = 0
# #     ask_volume_5: float = 0
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #
# #
# # @dataclass
# # class BarData(DataBase):
# #     """
# #     Candlestick bar data of a certain trading period.
# #     """
# #
# #     symbol: str
# #     exchange: Exchange
# #     datetime: datetime
# #
# #     interval: Interval = None
# #     volume: float = 0
# #     open_interest: float = 0
# #     open_price: float = 0
# #     high_price: float = 0
# #     low_price: float = 0
# #     close_price: float = 0
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #
# #
# # @dataclass
# # class OrderData(DataBase):
# #     """
# #     Order data contains information for tracking lastest status
# #     of a specific order.
# #     """
# #
# #     symbol: str
# #     exchange: Exchange
# #     orderid: str
# #
# #     type: OrderType = OrderType.LIMIT
# #     direction: Direction = None
# #     offset: Offset = Offset.NONE
# #     price: float = 0
# #     volume: float = 0
# #     traded: float = 0
# #     status: Status = Status.SUBMITTING
# #     datetime: datetime = None
# #     reference: str = ""
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #         self.e_orderid = f"{self.name}.{self.orderid}"
# #
# #     def is_active(self) -> bool:
# #         """
# #         Check if the order is active.
# #         """
# #         if self.status in ACTIVE_STATUSES:
# #             return True
# #         else:
# #             return False
# #
# #     def create_cancel_request(self) -> "CancelRequest":
# #         """
# #         Create cancel request object from order.
# #         """
# #         req = CancelRequest(
# #             orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
# #         )
# #         return req
# #
# #
# # @dataclass
# # class TradeData(DataBase):
# #     """
# #     Trade data contains information of a fill of an order. One order
# #     can have several trade fills.
# #     """
# #
# #     symbol: str
# #     exchange: Exchange
# #     orderid: str
# #     tradeid: str
# #     direction: Direction = None
# #
# #     offset: Offset = Offset.NONE
# #     price: float = 0
# #     volume: float = 0
# #     datetime: datetime = None
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #         self.e_orderid = f"{self.name}.{self.orderid}"
# #         self.e_tradeid = f"{self.name}.{self.tradeid}"
# #
# #
# # @dataclass
# # class StopOrder(object):
# #     e_symbol: str
# #     direction: Direction
# #     offset: Offset
# #     price: float
# #     volume: float
# #     stop_orderid: str
# #     strategy_name: str
# #     lock: bool = False
# #     vt_orderids: list = field(default_factory=list)
# #     status: StopOrderStatus = StopOrderStatus.WAITING
# #
# #
# # @dataclass
# # class PositionData(DataBase):
# #     """
# #     Positon data is used for tracking each individual position holding.
# #     """
# #
# #     symbol: str
# #     exchange: Exchange
# #     direction: Direction
# #
# #     volume: float = 0
# #     frozen: float = 0
# #     price: float = 0
# #     pnl: float = 0
# #     yd_volume: float = 0
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #         self.e_positionid = f"{self.e_symbol}.{self.direction.value}"
# #
# #
# # @dataclass
# # class AccountData(DataBase):
# #     """
# #     Account data contains information about balance, frozen and
# #     available.
# #     """
# #
# #     accountid: str
# #
# #     balance: float = 0
# #     frozen: float = 0
# #
# #     def __post_init__(self):
# #         """"""
# #         self.available = self.balance - self.frozen
# #         self.e_accountid = f"{self.name}.{self.accountid}"
# #
# #
# # @dataclass
# # class LogData(DataBase):
# #     """
# #     Log data is used for recording log messages on GUI or in log files.
# #     """
# #
# #     msg: str
# #     level: int = INFO
# #
# #     def __post_init__(self):
# #         """"""
# #         self.time = datetime.now()
# #
# #
# # @dataclass
# # class ContractData(DataBase):
# #     """
# #     Contract data contains basic information about each contract traded.
# #     """
# #
# #     symbol: str
# #     exchange: Exchange
# #     name: str
# #     product: Product
# #     size: int
# #     pricetick: float
# #
# #     min_volume: float = 1  # minimum trading volume of the contract
# #     stop_supported: bool = False  # whether server supports stop order
# #     net_position: bool = False  # whether gateway uses net position volume
# #     history_data: bool = False  # whether gateway provides bar history data
# #
# #     option_strike: float = 0
# #     option_underlying: str = ""  # vt_symbol of underlying contract
# #     option_type: OptionType = None
# #     option_expiry: datetime = None
# #     option_portfolio: str = ""
# #     option_index: str = ""  # for identifying options with same strike price
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #
# #
# # @dataclass
# # class SubscribeRequest:
# #     """
# #     Request sending to specific gateway for subscribing tick data update.
# #     """
# #
# #     symbol: str
# #     exchange: Exchange
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #
# #
# # @dataclass
# # class OrderRequest:
# #     """
# #     Request sending to specific gateway for creating a new order.
# #     """
# #
# #     symbol: str
# #     exchange: Exchange
# #     direction: Direction
# #     type: OrderType
# #     volume: float
# #     price: float = 0
# #     offset: Offset = Offset.NONE
# #     reference: str = ""
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #
# #     def create_order_data(self, orderid: str, source_name: str) -> OrderData:
# #         """
# #         Create order data from request.
# #         """
# #         order = OrderData(
# #             symbol=self.symbol,
# #             exchange=self.exchange,
# #             orderid=orderid,
# #             type=self.type,
# #             direction=self.direction,
# #             offset=self.offset,
# #             price=self.price,
# #             volume=self.volume,
# #             reference=self.reference,
# #             source_name=source_name,
# #         )
# #         return order
# #
# #
# # @dataclass
# # class CancelRequest:
# #     """
# #     Request sending to specific gateway for canceling an existing order.
# #     """
# #
# #     orderid: str
# #     symbol: str
# #     exchange: Exchange
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #
# #
# # @dataclass
# # class HistoryRequest:
# #     """
# #     Request sending to specific gateway for querying history data.
# #     """
# #
# #     symbol: str
# #     exchange: Exchange
# #     start: datetime
# #     end: datetime = None
# #     interval: Interval = None
# #
# #     def __post_init__(self):
# #         """"""
# #         self.e_symbol = f"{self.symbol}.{self.exchange.value}"
# #
# 1
if __name__ == '__main__':
    c = SimuTrade()
    print()
