import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from strategy.define_enum_constant import (TradeDirection, OrderStatus, TradeParam)


@dataclass
class Orders:
    multiplied_trailing_stop: float = None
    trailing_stop_value: float = None  # 用新的high或者low * multiplied_trailing_stop来更新
    multiplier: float = None  # 计算order的last_extreme_bar所在那天的5日day range均值/30个交易日5日day range均值的均值
    is_alive: OrderStatus = OrderStatus.NOT_OPENED  # 记录订单是否是（已经开仓未平仓）
    open_direction: TradeDirection = TradeDirection.NONE
    entry_idx: int = -1
    EntryTime: datetime = None  # 注意这边已经改成datetime类了
    EntryPrice: float = 0.0
    Lots: float = 0.0
    exit_index: int = -1
    ExitTime: datetime = None
    ExitPrice: float = 0.0
    # 下面两个是用来计算后面的浮盈和浮亏的
    high: float = 0.0
    low: float = np.inf
    floating_profit: float = 0.0
    floating_loss: float = 0.0
    # 后面需要的话可以记录是在第几根bar出现的最大浮盈和浮亏
    high_bar: int = 0
    low_bar: int = 0
    profit: float = 0.0
    # 记录前zzp的idx
    last_zzp_index: int = None
    last_zzp_value: float = None
    # 记录实时更新的止损线
    stoploss_line: float = None
    # 记录订单触发ema21需要达到的触及价格
    open_ema_price_threshold: float = None
    is_open_threshold: int = 0

    # def __init__(self, trailing_stop_num: float):
    #     self.trailing_stop_num = trailing_stop_num
    #     self.is_alive: OrderStatus = OrderStatus.NOT_OPENED  # 记录订单是否是（已经开仓未平仓）
    #     self.open_direction: TradeDirection = TradeDirection.NONE
    #     self.entry_idx: int = -1
    #     self.EntryTime: datetime = None  # 注意这边已经改成datetime类了
    #     self.EntryPrice: float = 0.0
    #     self.Lots: float = 0.0
    #     self.exit_index: int = -1
    #     self.ExitTime: datetime = None
    #     self.ExitPrice: float = 0.0
    #     # 下面两个是用来计算后面的浮盈和浮亏的
    #     self.high: float = 0.0
    #     self.low: float = np.inf
    #     self.floating_profit: float = 0.0
    #     self.floating_loss: float = 0.0
    #     # 后面需要的话可以记录是在第几根bar出现的最大浮盈和浮亏
    #     self.high_bar: int = 0
    #     self.low_bar: int = 0

    def __str__(self):
        return (f'trade: open: {self.open_direction} {self.EntryPrice}'
                f'{self.Lots} - finished={self.is_finished}')

    __repr__ = __str__

    # 记录当前的订单是否已经完成
    @property
    def is_finished(self) -> bool:
        return self.ExitTime is not None

    # Todo 怎么能让set函数用到呢，最后没用到
    def set_open_info(self, direction_, idx, price, lots, datetime_=None):
        """添加进场信息"""
        self.is_alive: OrderStatus = OrderStatus.IS_ALIVE
        self.open_direction = direction_
        self.entry_idx = idx
        self.EntryTime = datetime_
        self.EntryPrice = price
        self.Lots = lots

    def set_close_info(self, idx, price, datetime_=None):
        """添加离场信息"""
        self.is_alive: OrderStatus = OrderStatus.FLATTEN
        self.exit_index = idx
        self.ExitTime = datetime_
        self.ExitPrice = price

    def update_floating_profit_and_loss(self, new_high, new_low):

        if new_high > self.high:
            self.high = new_high
            self.high_bar += 1

        if new_low < self.low:
            self.low = new_low
            self.low_bar += 1

        if self.open_direction == TradeDirection.LONG:
            self.floating_profit = self.high - self.EntryPrice
            self.floating_loss = self.EntryPrice - self.low
        else:
            self.floating_profit = self.EntryPrice - self.low
            self.floating_loss = self.high - self.EntryPrice

    # Todo 下面要改成60个交易日 俺做不到
    def record_multiplied_trailing_stop_percentage(self, df_zigzag, trailing_stop_num, current_date, num=40):

        # entry_time = self.EntryTime  # datetime格式
        # entry_date = pd.to_datetime(datetime.strftime(entry_time, '%Y-%m-%d'))  # datetime格式
        # 把df_zigzag中的datetime列转化为timestamp格式
        df_zigzag['datetime'] = pd.to_datetime(df_zigzag['datetime'])
        # 记录当前日期
        startdate = current_date - timedelta(days=num)
        df_series = \
            df_zigzag[(df_zigzag.datetime >= startdate) & (df_zigzag.datetime < current_date)].groupby(['m_Date'])[
                'day_range'].max()
        if current_date < df_zigzag['datetime'].loc[4]:#current_date < pd.to_datetime('2018/1/6'):  # 这是因为前几天是获取不到前五日的均值的，需要人为修改日期，要保证mean_list中有至少5个数
            self.multiplied_trailing_stop = trailing_stop_num
            return self.multiplied_trailing_stop
        else:
            mean_list = df_series.rolling(5).mean()
            mean_avg_before, today_used_avg = mean_list.mean(), mean_list[-1]
            self.multiplied_trailing_stop = (today_used_avg / mean_avg_before) * trailing_stop_num
            return self.multiplied_trailing_stop

    def calc_profit(self):
        profit = self.ExitPrice - self.EntryPrice
        if self.open_direction == TradeDirection.SHORT:
            profit = -profit
        net_profit = (profit - TradeParam.COMMISSION_AND_SLIPPAGE.value) * self.Lots
        self.profit = net_profit

    def result_as_series(self):
        # return pd.Series(asdict(self)) #如果是需要输出所有变量的话，用这个比较方便，如果要自定义输出变量，用下面这种方式
        # 对于只需要获取进场信息的
        # return pd.Series({'entry_idx': self.entry_idx,
        #                   'EntryTime': self.EntryTime})
        #需要获取进场和出场的详细信息
        return pd.Series({'entry_idx': self.entry_idx,
                          'EntryTime': self.EntryTime,
                          'EntryPrice': self.EntryPrice,
                          'Direction': self.open_direction,
                          'exit_idx': self.exit_index,
                          'ExitTime': self.ExitTime,
                          'ExitPrice': self.ExitPrice,
                          'Lots': self.Lots,
                          'multiplier': self.multiplier,
                          'Commissions_and_slippage': TradeParam.COMMISSION_AND_SLIPPAGE.value,
                          'net_profit': self.profit,
                          'floating_profit': self.floating_profit,
                          'floating_loss': self.floating_loss,
                          'high_bar': self.high_bar,
                          'low_bar': self.low_bar,
                          'last_zzp_value': self.last_zzp_value})


if __name__ == '__main__':
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210301_{}.csv".format(
            0.5),
        index_col=0)
    df_all_zzg = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
            df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
    df_all_zzg["bar_num"] = df_all_zzg.index
    df_all_zzg = df_all_zzg.reset_index(drop=True)
    test_order = Orders(EntryTime=pd.to_datetime('2018-3-16'))
    list = []
    list.append(test_order)
    his_table = pd.DataFrame()
    for items in list:
        items.record_multiplied_trailing_stop_percentage(df_all_zzg, 0.0005, current_date=pd.to_datetime(
            '2018-12-14'))  # 如果重新打开注释，需要加入current date参数
        ts = items.multiplied_trailing_stop
        print(ts)
        sn = items.result_as_series()
        his_table.append(sn, ignore_index=True)
        print(his_table)
        list.remove(items)
        print(len(list))
