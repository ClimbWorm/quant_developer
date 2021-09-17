#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# General constant string
# Author: Tayii
# Data : 2021/1/18
# ----------------------------------------------------
from dataclasses import dataclass
from enum import Enum

# 标记值 所有的bar都处理完了
ALL_BAR_PROCESSED = -1


@dataclass
class TickPara:
    """交易对 tick相关参数"""
    usd_per_tick: float
    tick_size: float


SymbolTickPara = {
    'YM': TickPara(5.0, 1.0),  # 道琼斯
    '6B': TickPara(6.25, 0.0001),  # 英镑
    '6E': TickPara(6.25, 0.00005),  # 欧元
    'ES': TickPara(12.5, 0.25),  # 标普
}


class MProcess(Enum):
    """
    主控流程
    """
    BACK_TESTING = 'back_testing'
    TRADING = 'trading'


# 下面的是回测用 类型 ========================

class SimTradeStatus(Enum):
    """模拟交易状态（简单一点，只有开仓和持仓，没有已下单为成交等等其他状态）"""
    FLATTEN = "没有交易"  # 还未开仓/已平仓
    LONG_OPENED = "多单已开"  # 已开仓未平仓
    SHORT_OPENED = "空单已开"  # 已开仓未平仓


class OpenMax(Enum):
    """开仓限制"""
    LONG_1_OR_SHORT_1 = 1  # 只能选择一个方向 开一单
    LONG_1_SHORT_1 = 2
    LONG_n_OR_SHORT_n = 3
    LONG_n_SHORT_n = 4  # 多空均可开n单


class StopOrderStatus(Enum):
    WAITING = "等待中"
    CANCELLED = "已撤销"
    TRIGGERED = "已触发"


class EngineType(Enum):
    LIVE = "实盘"
    BACKTESTING = "回测"


class BacktestingMode(Enum):
    BAR = 1
    TICK = 2


# 下面的是交易数据类型 ========================

class Direction(Enum):
    """
    Direction of order/trade/position.
    """
    LONG = "多头"
    SHORT = "空头"
    NONE = "无"

    @classmethod
    def is_opposite(cls, d1, d2):
        """是否是相反反向"""
        return True if ((d1 == Direction.LONG and d2 == Direction.SHORT)
                        or (d1 == Direction.SHORT and d2 == Direction.LONG)) else False

    @classmethod
    def opposite(cls, d1):
        """取相反方向"""
        return d1 if d1 == Direction.NONE else (
            Direction.LONG if d1 == Direction.SHORT else Direction.SHORT)


class Sec(Enum):
    """
    Interval of bar data.
    """
    TICK = 1  # 假设1秒 主要用来判断
    MIN1 = 1 * 60
    MIN2 = 2 * 60
    MIN3 = 3 * 60
    MIN5 = 5 * 60
    MIN10 = 10 * 60
    MIN15 = 15 * 60
    MIN30 = 30 * 60
    HOUR1 = 1 * 60 * 60


class Offset(Enum):
    """
    Offset of order/trade.
    """
    NONE = ""
    OPEN = "开"
    CLOSE = "平"
    CLOSETODAY = "平今"
    CLOSEYESTERDAY = "平昨"


class Status(Enum):
    """
    Order status.
    """
    SUBMITTING = "提交中"
    NOTTRADED = "未成交"
    PARTTRADED = "部分成交"
    ALLTRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"


class Product(Enum):
    """
    Product class.
    """
    EQUITY = "股票"
    FUTURES = "期货"
    OPTION = "期权"
    INDEX = "指数"
    FOREX = "外汇"
    SPOT = "现货"
    ETF = "ETF"
    BOND = "债券"
    WARRANT = "权证"
    SPREAD = "价差"
    FUND = "基金"


class OrderType(Enum):
    """
    Order type.
    """
    LIMIT = "限价"
    MARKET = "市价"
    STOP = "STOP"
    FAK = "FAK"
    FOK = "FOK"
    RFQ = "询价"


class OptionType(Enum):
    """
    Option type.
    """
    CALL = "看涨期权"
    PUT = "看跌期权"


class Exchange(Enum):
    """
    Exchange.
    """
    # Chinese
    CFFEX = "CFFEX"  # China Financial Futures Exchange
    SHFE = "SHFE"  # Shanghai Futures Exchange
    CZCE = "CZCE"  # Zhengzhou Commodity Exchange
    DCE = "DCE"  # Dalian Commodity Exchange
    INE = "INE"  # Shanghai International Energy Exchange
    SSE = "SSE"  # Shanghai Stock Exchange
    SZSE = "SZSE"  # Shenzhen Stock Exchange
    SGE = "SGE"  # Shanghai Gold Exchange
    WXE = "WXE"  # Wuxi Steel Exchange
    CFETS = "CFETS"  # China Foreign Exchange Trade System

    # Global
    SMART = "SMART"  # Smart Router for US stocks
    NYSE = "NYSE"  # New York Stock Exchnage
    NASDAQ = "NASDAQ"  # Nasdaq Exchange
    ARCA = "ARCA"  # ARCA Exchange
    EDGEA = "EDGEA"  # Direct Edge Exchange
    ISLAND = "ISLAND"  # Nasdaq Island ECN
    BATS = "BATS"  # Bats Global Markets
    IEX = "IEX"  # The Investors Exchange
    NYMEX = "NYMEX"  # New York Mercantile Exchange
    COMEX = "COMEX"  # COMEX of CME
    GLOBEX = "GLOBEX"  # Globex of CME
    IDEALPRO = "IDEALPRO"  # Forex ECN of Interactive Brokers
    CME = "CME"  # Chicago Mercantile Exchange
    ICE = "ICE"  # Intercontinental Exchange
    SEHK = "SEHK"  # Stock Exchange of Hong Kong
    HKFE = "HKFE"  # Hong Kong Futures Exchange
    HKSE = "HKSE"  # Hong Kong Stock Exchange
    SGX = "SGX"  # Singapore Global Exchange
    CBOT = "CBT"  # Chicago Board of Trade
    CBOE = "CBOE"  # Chicago Board Options Exchange
    CFE = "CFE"  # CBOE Futures Exchange
    DME = "DME"  # Dubai Mercantile Exchange
    EUREX = "EUX"  # Eurex Exchange
    APEX = "APEX"  # Asia Pacific Exchange
    LME = "LME"  # London Metal Exchange
    BMD = "BMD"  # Bursa Malaysia Derivatives
    TOCOM = "TOCOM"  # Tokyo Commodity Exchange
    EUNX = "EUNX"  # Euronext Exchange
    KRX = "KRX"  # Korean Exchange
    OTC = "OTC"  # OTC Forex Broker
    IBKRATS = "IBKRATS"  # Paper Trading Exchange of IB

    # CryptoCurrency
    BITMEX = "BITMEX"
    OKEX = "OKEX"
    HUOBI = "HUOBI"
    BITFINEX = "BITFINEX"
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"  # bybit.com
    COINBASE = "COINBASE"
    DERIBIT = "DERIBIT"
    GATEIO = "GATEIO"
    BITSTAMP = "BITSTAMP"

    # Special Function
    LOCAL = "LOCAL"  # For local generated data


class Currency(Enum):
    """
    Currency.
    """
    USD = "USD"
    HKD = "HKD"
    CNY = "CNY"


class Interval(Enum):
    """
    Interval of bar data.
    """
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "d"
    WEEKLY = "w"
    TICK = "tick"
