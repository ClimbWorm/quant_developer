from enum import Enum


class TradeDirection(Enum):
    LONG = 'long'
    SHORT = 'short'
    NONE = None


class OrderStatus(Enum):
    IS_ALIVE = "opened_and_not_close"
    FLATTEN = "opened_and_closed"
    NOT_OPENED = "not_opened"


class TradeParam(Enum):
    COMMISSION_AND_SLIPPAGE: float = 3.0
