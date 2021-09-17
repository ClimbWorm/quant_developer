import time

import numpy as np
import pandas as pd
import os, sys
import multiprocessing
import talib
from dataclasses import dataclass, field
from strategy.define_enum_constant import *
from strategy.functionsforATRRegression import *
from strategy.order_management import *
from tools.backtesting_backup.get_specific_enter_price import *
from rmt_env import RemoteEnv

epsilon = 1e-5


@dataclass
class Strategy(object):

    def __init__(self, zzg_num, ema_num, trailing_stop_num, open_threshold_multiplier, path):

        # path = "F:\\pythonHistoricalTesting\\pythonHistoricalTesting\\code_generated_csv\\ema21_trailing_stop\\"
        # path = '..\\backtesting_data\\'

        filename = "df_backtesting_all_rsi_ohlc_avg_{}.csv".format(zzg_num)

        try:
            self.df_backtesting = pd.read_csv(path + filename, index_col=0)
        except Exception as e:
            filename = "df_backtesting_all_rsi_ohlc_avg_{}.csv".format(int(zzg_num))
            self.df_backtesting = pd.read_csv(path + filename, index_col=0)


        # self.df_backtesting:pd.DataFrame = pd.read_csv(path+filename,index_col=0)
        # self.df_backtesting: pd.DataFrame = self.process_df_backtesting()  # 回测数据
        self.zzg_num = zzg_num
        self.ema_num = ema_num
        self.trailing_stop_num = trailing_stop_num
        self.is_alive_orders = []  # list中为一个个order的实例
        # 记录历史记录
        self.his_table = pd.DataFrame(
            columns=['entry_idx', 'EntryTime', 'EntryPrice', 'Direction', 'exit_idx', 'ExitTime', 'ExitPrice', 'Lots',
                     'multiplier', 'Commissions_and_slippage', 'net_profit', 'floating_profit',
                     'floating_loss', 'high_bar', 'low_bar', 'last_zzp_value']
        )
        self.df_Day = self.generate_df_Day(path + '/YMH21-CBOT_day_from_2012_01_01.txt')
        # 寻找计算概率的表
        self.df_D = pd.DataFrame()
        self.df_all_zzg = self.generate_zzg_table_include_once_and_final(path)
        self.trade_params = {'max_lots': 10}

        # 与zzp相关的
        self.last_zzg_bar_Idx, self.last_extreme_bar_low_or_high, self.last_extreme_value, self.last_extreme_bar_threshold \
            = lastExtremeBar(0, self.df_all_zzg)
        # 打开阈值，把止损线设置为ema21前需要达到的阈值乘数
        self.open_threshold_multiplier = open_threshold_multiplier

        # 记录已经遍历到的日期，目的是为了减少更新df_D的运算量
        self.already_passed_date = []

        # 回测需要的超参数 list
        # self.hyper_parameter = [{"zzg_num": zzg_num, "trailing_stop_num": trailing_stop_num, }
        #                         for zzg_num in [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
        #                         for trailing_stop_num in np.arange(0.0005, 0.01, 0.00025)]

    def add_ema(self, df_backtesting, period=21):
        df_backtesting[f'ema_{period}'] = talib.EMA(df_backtesting.Close,
                                                    timeperiod=period)  # .shift(1) ema的计算是包括当前bar的
        return df_backtesting

    # def add_bollinger_bands(self, df_backtesting, period=55):
    #     H_line, M_line, L_line = talib.BBANDS(df_backtesting.Close, timeperiod=period, nbdevup=2, nbdevdn=2,
    #                                           matype=1).shift(1)  # matype=1表示ema
    #     df_backtesting[f'H_line_{period}'] = H_line
    #     df_backtesting[f'M_line_{period}'] = M_line
    #     df_backtesting[f'L_line_{period}'] = L_line
    #     return df_backtesting

    def add_rsi(self, df_backtesting, df_60min, period=60, input_data=1):
        """input_data：1 表示close 2 表示用开高低收的均值
        最后一个close或者开高低收的均值取当前的15min的bar，之前的数据都取60min的"""
        rsi_list = []
        for t in range(len(df_backtesting)):
            time = pd.to_datetime(df_backtesting.Time.loc[t])
            t_ = df_60min.loc[df_60min.Time == df_backtesting.Time.loc[t]].index[0]  # 60min表中的index
            if time.strftime("%M") == "45":
                # 包括当前bar，只要在df_60min中取15个数即可
                if input_data == 1:
                    try:
                        rsi_sub = talib.RSI(df_60min.Close.loc[t_ - 14:t_], timeperiod=14)[t_]
                    except Exception as e:
                        rsi_sub = np.nan
                else:
                    try:
                        avg_OHLC = np.mean(
                            [df_60min.Open.loc[t_ - 14:t_], df_60min.High.loc[t_ - 14:t_], df_60min.Low.loc[t_ - 14:t_],
                             df_60min.Close.loc[t_ - 14:t_]], axis=0)
                        rsi_sub = talib.RSI(pd.Series(avg_OHLC))[14]
                    except Exception as e:
                        rsi_sub = np.nan
            else:
                # 当前bar的数值（在df_backtesting这个15min的表中找，再往前取14个数）
                if input_data == 1:
                    try:
                        data_series = df_60min.Close.loc[t_ - 14:t_ - 1].tolist() + [df_backtesting.Close.loc[t]]
                        rsi_sub = talib.RSI(pd.Series(data_series))[14]
                    except Exception as e:
                        rsi_sub = np.nan
                else:
                    try:
                        avg_OHLC = np.mean([df_60min.Open.loc[t_ - 14:t_ - 1], df_60min.High.loc[t_ - 14:t_ - 1],
                                            df_60min.Low.loc[t_ - 14:t_ - 1], df_60min.Close.loc[t_ - 14:t_ - 1]],
                                           axis=0)
                        data_series = avg_OHLC.tolist() + [np.mean(
                            [df_backtesting.Open.loc[t], df_backtesting.High.loc[t], df_backtesting.Low.loc[t],
                             df_backtesting.Close.loc[t]])]
                        rsi_sub = talib.RSI(pd.Series(data_series))[14]
                    except Exception as e:
                        rsi_sub = np.nan
            rsi_list.append(rsi_sub)
        df_backtesting[f'rsi_{period}_min'] = rsi_list
        return df_backtesting

    # def process_df_backtesting(self):
    # 更具体的在get only entry info中
    #     df_backtesting = pd.read_csv(
    #         '../backtesting_data/back.csv')
    #     # df_60min = pd.read_csv()
    #     df_backtesting['Time'] = pd.to_datetime(df_backtesting['Time'])
    #     df_backtesting = get_atr_breakout_price(df_backtesting, 1.618)
    #     df_backtesting = self.add_ema(df_backtesting)
    #     # df_backtesting = self.add_bollinger_bands(df_backtesting)
    #     # df_backtesting = self.add_rsi(df_backtesting,df_60min)
    #     # 做数据清洗
    #     df_backtesting = df_backtesting[
    #         ~df_backtesting.isin([np.nan, np.inf, -np.inf]).any(1)]  # 这边不要reset_index，后面注意loc和iloc的使用
    #     # df_backtesting.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\cleaned_data.csv')
    #     # df_backtesting.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\df_backtesting_rsi_close.csv')
    #     return df_backtesting

    def generate_zzg_table_include_once_and_final(self, path):
        zzg_num = self.zzg_num
        try:
            df_zigzag = pd.read_csv(path + "/zigzag_20210331_{}.csv".format(zzg_num), index_col=0)
        except Exception as e:
            df_zigzag = pd.read_csv(path + "/zigzag_20210331_{}.csv".format(int(zzg_num)), index_col=0)
        df_all_zzg = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
                df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
        df_all_zzg["bar_num"] = df_all_zzg.index
        df_all_zzg = df_all_zzg.reset_index(drop=True)
        df_all_zzg['m_Date'] = pd.to_datetime(df_all_zzg['m_Date'], format="%Y-%m-%d")
        return df_all_zzg

    def update_df_D(self, current_date_):
        """时间都统一为时间格式，而不是字符串格式"""
        df_Day = self.df_Day
        if len(self.already_passed_date) == 0:
            self.df_D = generate_df_D(current_date_, df_Day)
            # Todo 对于不需要获取出场信息的或者不需要用到trailing stop的，下面这一行需要打开
            self.already_passed_date.append(current_date_)
        else:
            if current_date_ != self.already_passed_date[-1]:
                self.df_D = generate_df_D(current_date_, df_Day)
                # Todo 对于不需要获取出场信息的或者不需要用到trailing stop的，下面这一行需要打开
                self.already_passed_date.append(current_date_)

    def set_entry_lots(self, i_):
        return SetLots(self.df_backtesting.loc[i_].last_zscore, self.df_backtesting.loc[i_].realtime_zscore, self.df_D,
                       miniLots=0, maxLotsLimit=10)

    @staticmethod
    def generate_df_Day(df_txt):
        # df_Day = pd.read_csv(r'..\backtesting_data\YMH21-CBOT_day_from_2012_01_01.txt')
        df_Day = pd.read_csv(df_txt)
        df_Day = df_Day.loc[:,
                 ['Date', ' AskVolume', ' BidVolume', ' High', ' Low', ' NumberOfTrades', ' Volume', ' Open', ' Last']]
        df_Day.columns = ["DateTag", "AskVolume", "BidVolume", "High", "Low", "NumberOfTrades", "Volume", "Open",
                          "Close"]
        df_Day['DateTag'] = pd.to_datetime(df_Day['DateTag'])
        return df_Day

    def judge_open_direction(self):
        if (self.last_extreme_bar_low_or_high == "ZZPT.ONCE_HIGH") | (
                self.last_extreme_bar_low_or_high == "ZZPT.HIGH"):
            return TradeDirection.SHORT
        elif (self.last_extreme_bar_low_or_high == "ZZPT.ONCE_LOW") | (
                self.last_extreme_bar_low_or_high == "ZZPT.LOW"):
            return TradeDirection.LONG
        else:
            pass


    def can_open(self, i_, open_direction_) -> bool:
        if self.can_be_optimized_entry(i_):
            if open_direction_ == TradeDirection.SHORT:
                return (self.df_backtesting.loc[i_].HA_Close - self.df_backtesting.loc[i_].HA_Open) < 0

            else:
                return (self.df_backtesting.loc[i_].HA_Close - self.df_backtesting.loc[i_].HA_Open) > 0

        else:  # 没有突破
            if open_direction_ == TradeDirection.SHORT:
                return (self.df_backtesting.loc[i_ - 1].HA_Close - self.df_backtesting.loc[i_ - 1].HA_Open) < 0

            else:
                return (self.df_backtesting.loc[i_ - 1].HA_Close - self.df_backtesting.loc[i_ - 1].HA_Open) > 0

    # def can_open(self, i_, open_direction_) -> bool:
    #     if self.can_be_optimized_entry(i_):
    #         if open_direction_ == TradeDirection.SHORT:
    #             if (self.df_backtesting.loc[i_].HA_Close - self.df_backtesting.loc[i_].HA_Open) < 0:
    #                 if (80 < self.df_backtesting.loc[i_].rsi_60_min < 100)|(self.df_backtesting.loc[i_].RSIShortgodown == 1)|(self.df_backtesting.loc[i_].RSI_TopDivergence == 1):
    #                     return True
    #             return False
    #         else:
    #             if (self.df_backtesting.loc[i_].HA_Close - self.df_backtesting.loc[i_].HA_Open) > 0:
    #                 if (0 < self.df_backtesting.loc[i_].rsi_60_min < 20)|(self.df_backtesting.loc[i_].RSIShortgoup == 1)|(self.df_backtesting.loc[i_].RSI_BottomDivergence == 1):
    #                     return True
    #             return False
    #     else:  # 没有突破
    #         if open_direction_ == TradeDirection.SHORT:
    #             if (self.df_backtesting.loc[i_ - 1].HA_Close - self.df_backtesting.loc[i_ - 1].HA_Open) < 0:
    #                 if (80 < self.df_backtesting.loc[i_-1].rsi_60_min < 100)|(self.df_backtesting.loc[i_-1].RSIShortgodown == 1)|(self.df_backtesting.loc[i_-1].RSI_TopDivergence == 1):
    #                     return True
    #             return False
    #         else:
    #             if (self.df_backtesting.loc[i_ - 1].HA_Close - self.df_backtesting.loc[i_ - 1].HA_Open) > 0:
    #                 if (0 < self.df_backtesting.loc[i_-1].rsi_60_min < 20)|(self.df_backtesting.loc[i_-1].RSIShortgoup == 1)|(self.df_backtesting.loc[i_-1].RSI_BottomDivergence == 1):
    #                     return True
    #             return False

    def is_limit_open_by_same_previous_zzp(self, last_zzg_bar_Idx_, is_alive_orders_, open_direction_) -> bool:
        if len(is_alive_orders_) == 0:
            return True
        for order_ in is_alive_orders_:  # Todo 这边对于不需要获取离场信息的，就只遍历最后加入的30个即可
            if order_.last_zzp_index == last_zzg_bar_Idx_:
                if order_.open_direction == open_direction_:
                    return False
        return True

    def can_be_optimized_entry(self, i_) -> bool:
        if self.df_backtesting.loc[i_].is_atr_breakout:
            return True
        else:
            return False

    def set_specific_entry_price(self, i_, open_direction_):
        if self.can_be_optimized_entry(i_):
            if open_direction_ == TradeDirection.SHORT:
                if self.df_backtesting.loc[i_].Close < self.df_backtesting.loc[i_].Open:  # 当前这根bar是阴线，，，Todo 总觉得这里有未来函数，因为只有当这根bar走完了，才知道这根bar是不是阴线
                    return self.df_backtesting.loc[i_].previous_close_minus_multiple_range  # 这边感觉可能是优化了，也可能是缩小了利润
                else:
                    return self.df_backtesting.loc[i_].Close  # Todo 这边有突破但是不是阴线，那就close进？原本这里写的是Open
            else:
                if self.df_backtesting.loc[i_].Close > self.df_backtesting.loc[i_].Open:  # 当前这根bar是阳线
                    return self.df_backtesting.loc[i_].previous_close_plus_multiple_range
                else:
                    return self.df_backtesting.loc[i_].Close  ###
        else:
            return self.df_backtesting.loc[i_].Open

    def new_an_order_object(self, open_direction_, entry_idx_, EntryTime_, EntryPrice_, Lots_, last_zzp_index_,
                            last_zzp_value_, stoploss_line_, open_ema_price_threshold_,
                            order_status_=OrderStatus.IS_ALIVE):
        _new_trade = Orders(is_alive=order_status_,
                            open_direction=open_direction_,
                            entry_idx=entry_idx_,
                            EntryTime=EntryTime_,
                            EntryPrice=EntryPrice_,
                            Lots=Lots_,
                            last_zzp_index=last_zzp_index_,
                            last_zzp_value=last_zzp_value_,
                            stoploss_line=stoploss_line_,
                            open_ema_price_threshold=open_ema_price_threshold_,
                            is_open_threshold=0)  # 实例化一个订单对象
        # print(self.df_all_zzg[self.df_all_zzg.bar_num == last_zzp_index_].m_Date.values)
        date_of_last_extreme_bar = pd.to_datetime(
            self.df_all_zzg[self.df_all_zzg.bar_num == last_zzp_index_].m_Date.values[0])
        _new_trade.multiplier = _new_trade.record_multiplied_trailing_stop_percentage(self.df_all_zzg,
                                                                                      self.trailing_stop_num,
                                                                                      date_of_last_extreme_bar) / self.trailing_stop_num

        self.is_alive_orders.append(_new_trade)
        # Todo 对于不需要获取最终离场信息的，还要打开下面这行
        # self.his_table = self.his_table.append(_new_trade.result_as_series(), ignore_index=True)

    def exit_order(self, i_, CurrentTime_):
        # 下面都是为离场做准备的更新信息
        for order in self.is_alive_orders:
            # 更新order的high,low,floating_profit,floating_loss,high_bar,low_bar信息
            high_updated, low_updated = 0, 0
            if self.df_backtesting.loc[i_].High > order.high:
                order.high = self.df_backtesting.loc[i_].High
                order.high_bar = i_
                high_updated = 1

            if self.df_backtesting.loc[i_].Low < order.low:
                order.low = self.df_backtesting.loc[i_].Low
                order.low_bar = i_
                low_updated = 1

            if order.open_direction == TradeDirection.SHORT:
                if low_updated:
                    order.floating_profit = order.EntryPrice - order.low

                if high_updated:
                    order.floating_loss = order.high - order.EntryPrice
            else:
                if high_updated:
                    order.floating_profit = order.high - order.EntryPrice

                if low_updated:
                    order.floating_loss = order.EntryPrice - order.low

            # —————————————————————原本逻辑中的trailing stop（根据实时更新的百分比来的）—————————————————————————————————
            # 更新multiplied_trailing_stop信息
            # if len(self.already_passed_date) == 0:
            #     order.record_multiplied_trailing_stop_percentage(self.df_all_zzg, self.trailing_stop_num,
            #                                                      current_date)
            #     self.already_passed_date.append(current_date) # Todo 这里有更新列表
            #
            # else:
            #     if current_date != self.already_passed_date[-1]:
            #         order.record_multiplied_trailing_stop_percentage(self.df_all_zzg, self.trailing_stop_num,
            #                                                          current_date)
            #         self.already_passed_date.append(current_date)  # Todo 这里有更新列表

            # 更新trailing stop value信息，遍历self.is_alive_orders中的订单
            # if order.open_direction == TradeDirection.SHORT:  # 如果是做空的订单，那么出现新low就要更新stop的值
            #     order.trailing_stop_value = order.low * (1 + order.multiplied_trailing_stop)
            # else:
            #     order.trailing_stop_value = order.high * (1 - order.multiplied_trailing_stop)

            # 处理止损离场
            # if order.open_direction == TradeDirection.SHORT:
            #     if order.high >= order.trailing_stop_value:  # 碰到止损线
            #         order.exit_index = i
            #         order.ExitTime = CurrentTime
            #         order.ExitPrice = order.trailing_stop_value
            #         order.is_alive = OrderStatus.FLATTEN
            # else:
            #     if order.low <= order.trailing_stop_value:
            #         order.exit_index = i
            #         order.ExitTime = CurrentTime
            #         order.ExitPrice = order.trailing_stop_value
            #         order.is_alive = OrderStatus.FLATTEN
            # ————————————————————————————原本的trailing stop逻辑结束——————————————————————————————————————————————

            # 更新stoploss_line
            # 止损逻辑：用close的值和止损线比较，碰到止损线用close离场
            # long：进场时止损线设置在前赋值点的low-epsilon
            # 当bar的high大于21日均线后，把止损线位置挪到entryprice-1tick
            # 当bar的high大于前赋值点 + 五日range的均值*（0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2）*（1,1.236,1.5,1.618,2），止损线就为ema21
            #
            # short: 进场时止损线设置在前赋值点的high + epsilon
            # 当bar的low小于21日均线后，把止损线位置挪到entryprice+1tick
            # 当bar的low小于前赋值点 - 五日range的均值*（0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2）*（1,1.236,1.5,1.618,2），止损线就为ema21
            if order.open_direction == TradeDirection.SHORT:
                if (self.df_backtesting.loc[i_ - 1].Low < self.df_backtesting.loc[i_ - 1][
                    f"ema_{int(self.ema_num)}"]) &(self.df_backtesting.loc[i_ - 2].Low > self.df_backtesting.loc[i_ - 2][
                    f"ema_{int(self.ema_num)}"])& (
                        # Todo 这边的Low后续可以改成close试试
                        order.is_open_threshold == 0):  # 还未开启ema21的移动止损
                    order.stoploss_line = order.EntryPrice + 1

                if (self.df_backtesting.loc[i_ - 1].Low < order.open_ema_price_threshold) & (self.df_backtesting.loc[i_ - 1].Low < self.df_backtesting.loc[i_ - 1][
                    f"ema_{int(self.ema_num)}"]) & (
                        order.is_open_threshold == 0):
                    order.stoploss_line = self.df_backtesting.loc[i_ - 1][f"ema_{int(self.ema_num)}"]
                    order.is_open_threshold = 1

                if order.is_open_threshold == 1:
                    order.stoploss_line = self.df_backtesting.loc[i_ - 1][f"ema_{int(self.ema_num)}"]
            else:
                # Todo i-1 high > i-1 ema i-2的high小于i-2 ema
                if (self.df_backtesting.loc[i_ - 1].High > self.df_backtesting.loc[i_ - 1][
                    f"ema_{int(self.ema_num)}"]) & (self.df_backtesting.loc[i_ - 2].High < self.df_backtesting.loc[i_ - 2][
                    f"ema_{int(self.ema_num)}"]) & (
                        # Todo 这边的High后续可以改成close试试
                        order.is_open_threshold == 0):  # 还未开启ema21的移动止损
                    order.stoploss_line = order.EntryPrice - 1
                # Todo i-1 high > i-1 ema
                if (self.df_backtesting.loc[i_ - 1].High > order.open_ema_price_threshold) & (self.df_backtesting.loc[i_ - 1].High > self.df_backtesting.loc[i_ - 1][
                    f"ema_{int(self.ema_num)}"]) & (
                        order.is_open_threshold == 0):
                    order.stoploss_line = self.df_backtesting.loc[i_ - 1][f"ema_{int(self.ema_num)}"]
                    order.is_open_threshold = 1

                if order.is_open_threshold == 1:
                    order.stoploss_line = self.df_backtesting.loc[i_ - 1][f"ema_{int(self.ema_num)}"]

            # 处理止损离场
            if order.open_direction == TradeDirection.SHORT:
                if self.df_backtesting.loc[i_].High >= order.stoploss_line:
                    order.exit_index = i_
                    order.ExitTime = CurrentTime_
                    order.ExitPrice = order.stoploss_line
                    # if order.stoploss_line >= order.EntryPrice:
                    #     order.ExitPrice = order.stoploss_line
                    # else:
                    #     order.ExitPrice = self.df_backtesting.loc[i_].Close
                    order.is_alive = OrderStatus.FLATTEN
            else:
                # if order.entry_idx == 614:
                #     print(i_,self.df_backtesting.loc[i_].Close <= order.stoploss_line,"Close：",self.df_backtesting.loc[i_].Close,"止损线：",order.stoploss_line,"\n",self.df_backtesting.loc[i_].Time)
                if self.df_backtesting.loc[i_].Low <= order.stoploss_line:
                    order.exit_index = i_
                    order.ExitTime = CurrentTime_
                    order.ExitPrice = order.stoploss_line
                    # if order.stoploss_line <= order.EntryPrice:  # 浮亏了
                    #     order.ExitPrice = order.stoploss_line
                    # else:
                    #     order.ExitPrice = self.df_backtesting.loc[i_].Close
                    order.is_alive = OrderStatus.FLATTEN

        # 处理已经平仓的订单
        for order in self.is_alive_orders:
            if order.is_alive == OrderStatus.FLATTEN:
                order.calc_profit()
                self.his_table = self.his_table.append(order.result_as_series(), ignore_index=True)
                self.is_alive_orders.remove(order)

    def main(self):
        """如果有同一个前zzp的同向订单限制再开同向订单，没有反向信号出现就平仓的逻辑"""
        traverse_list = set(self.df_backtesting.index) & set(self.df_backtesting.index + 1) & set(self.df_backtesting.index + 2)
        traverse_list = sorted(traverse_list)
        # print(traverse_list)
        for i in traverse_list:  # range(3000,e):# 因为前面有些信息缺失2000
            CurrentTime = pd.to_datetime(
                self.df_backtesting.loc[i].Time)  # Todo 如果df_backtesing是通过读入外部文件产生的，这边需要转化成时间形式，不然直接读就行
            # PreviousTime = self.df_backtesting.loc[i - 1].Time
            current_date = pd.to_datetime(CurrentTime.strftime("%Y-%m-%d"))

            # 计算当前bar距离上一个极值点的距离,获取到上一个极值点的类型，是高点还是低点及其数值
            self.last_zzg_bar_Idx, self.last_extreme_bar_low_or_high, self.last_extreme_value, self.last_extreme_bar_threshold = \
                lastExtremeBar(i, self.df_all_zzg)

            if self.last_zzg_bar_Idx == 0:
                continue

            distance = i - self.last_zzg_bar_Idx
            if ((self.zzg_num < 0.886) & (distance > 5)) | ((self.zzg_num >= 0.886) & (distance > 10)):
                # 这种情况下不需要判断开仓，但是仍然需要判断是否有离场机会
                # 判断离场
                self.exit_order(i, CurrentTime)
            else:
                self.update_df_D(current_date)  # 更新概率表
                open_direction = self.judge_open_direction()

                # 进场
                if self.can_open(i, open_direction):
                    if self.is_limit_open_by_same_previous_zzp(self.last_zzg_bar_Idx, self.is_alive_orders,
                                                               open_direction):
                        lots_sub = self.set_entry_lots(i)
                        entry_price_sub = self.set_specific_entry_price(i, open_direction)

                        if open_direction == TradeDirection.SHORT:
                            stoploss_line = self.last_extreme_value + epsilon
                            open_ema_price_threshold = self.last_extreme_value - self.last_extreme_bar_threshold * self.open_threshold_multiplier
                        else:
                            stoploss_line = self.last_extreme_value - epsilon
                            open_ema_price_threshold = self.last_extreme_value + self.last_extreme_bar_threshold * self.open_threshold_multiplier

                        self.new_an_order_object(open_direction_=open_direction, entry_idx_=i, EntryTime_=CurrentTime,
                                                 EntryPrice_=entry_price_sub, Lots_=lots_sub,
                                                 last_zzp_index_=self.last_zzg_bar_Idx,
                                                 last_zzp_value_=self.last_extreme_value, stoploss_line_=stoploss_line,
                                                 open_ema_price_threshold_=open_ema_price_threshold)
                # 判断离场
                self.exit_order(i, CurrentTime)

        return self.his_table


def save_result(zzg_num, ema_num, open_threshold_multiplier, save_result_path, csv_path):
    import matplotlib.pyplot as plt
    # path = r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_added_ema_differ\use_low_high_stopline_out\pnl_{}_{}_{}.png'.format(
    #             zzg_num,ema_num,open_threshold_multiplier)
    path_pic = os.path.join(save_result_path, 'pnl_{}_{}_{}.png'.format(zzg_num, ema_num, open_threshold_multiplier))
    path_csv = os.path.join(save_result_path, 'table_{}_{}_{}.csv'.format(zzg_num, ema_num, open_threshold_multiplier))
    if os.path.exists(path_pic):
        pass
    else:
        strategy = Strategy(zzg_num, ema_num, 1, open_threshold_multiplier,
                            csv_path)  # Todo 对于不需要获取最终结果的，或者没有用到trailing stop的，第二个参数写一个除0外的数就可以
        strategy.main()
        plt.figure(figsize=(20, 8))
        plt.plot(strategy.his_table.net_profit.cumsum().values)
        plt.title('pnl')
        plt.savefig(path_pic)
        plt.close()
        strategy.his_table.to_csv(path_csv)


# def generate_params_config():
#     arg_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1]
#     multiplier_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1,1.236, 1.5, 2]
#     arg_list1 = [1.236, 1.5, 2]
#     multiplier_list1 = [0.382, 0.5, 0.618, 0.782, 0.886, 1]
#     # 生成params_config
#     params_config = [[i, j] for i in arg_list for j in multiplier_list]
#     params_config1 = [[i, j] for i in arg_list1 for j in multiplier_list1]
#     # print(params_config1 + params_config)
#     pd.DataFrame(params_config1 + params_config, columns=['num', 'multiplier']).to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\params_config\params_config_ha_ema21_trailing_stop.csv')
#     return pd.DataFrame(params_config1 + params_config, columns=['num', 'multiplier'])
#
# def add_ema(zzg_num):
#     path = "F:\\pythonHistoricalTesting\\pythonHistoricalTesting\\code_generated_csv\\ema21_trailing_stop\\"
#     filename = "df_backtesting_all_rsi_ohlc_avg_{}.csv".format(zzg_num)
#     df_backtesting = pd.read_csv(path + filename, index_col=0)
#     for t in range(13,56):
#         df_backtesting[f'ema_{t}'] = talib.EMA(df_backtesting.Close,timeperiod=t)
#     df_backtesting.to_csv(path + filename)


if __name__ == '__main__':
    # 单个测试的代码
    # save_result_path = r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out'
    # save_result(0.382,13,0.1,save_result_path)
    # strategy = Strategy(1.236, 1,1)
    # strategy.main()
    # strategy.his_table.to_csv(
    #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\table_{}_{}.csv'.format(
    #         1.236,1))

    # 还需要修改文件保存的位置
    # import sys
    # import os
    # batch_id,cores,param_file,data_path = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
    # print(batch_id,cores,param_file)
    # cores = int(cores)
    # print(type(cores))
    # param_content = pd.read_csv(param_file,index_col=0)
    # zzg_num_list,ema_num_list,multiplier_list = param_content.zzg_num[:4], param_content.ema_num[:4], param_content.multiplier[:4]
    # pool = multiprocessing.Pool(cores)
    # for zzg_num,ema_num,multiplier in zip(zzg_num_list,ema_num_list,multiplier_list):
    #     pool.apply_async(save_result,(zzg_num,ema_num,multiplier,data_path))
    # pool.close()
    # pool.join()

    # for i in range(len(param_content)):
    #     zzg_num, ema_num, open_threshold_multiplier = param_content.loc[i]['zzg_num'],param_content.loc[i]['ema_num'],param_content.loc[i]['multiplier']
    #     save_result(zzg_num, ema_num, open_threshold_multiplier)
    # os.system('touch ' + str(batch_id))

    # pool = multiprocessing.Pool(10)
    # i = 1
    # for zzg_num in [0.382, 0.5, 0.618, 0.782, 0.886, 1,1.236, 1.5, 2]:
    #
    #     rst = pool.apply_async(add_ema,(zzg_num,))
    #     rst.get()
    #     print("i:",i)
    #     i += 1
    # pool.close()
    # pool.join()

    # df_params = generate_params_config()
    # print(df_params)
    # params = pd.read_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\params_config\params_config_ha_ema21_trailing_stop.csv',index_col=0)

    # 用close出有盈利的参数组合
    # param_list = [] #由元祖组成的list
    # his_table_sub = pd.read_csv(
    #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\rsi_all_added_ema_differ_drawdown_stat_using_close.csv')
    # for i,j,k in zip(his_table_sub.zzg_num, his_table_sub.ema_num,his_table_sub.multiplier):
    #     if i == float(1):
    #         param_list.append((int(i),j,k))
    #     else:
    #         param_list.append((i,j,k))
    # pool = multiprocessing.Pool(10)
    # i = 1
    # for params in param_list:
    #     pool.apply_async(save_result,params)
    #     print("i:", i)
    #     i += 1
    # pool.close()
    # pool.join()
    # print("=====================finished!===========================")
    # Todo 保存路径稍后再改
    save_result_path = r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\only_ha_ema_differ\use_low_high_stopline_out'

    # Todo 跑的时候注意can open那边的限制条件是否需要打开
    remote_env = RemoteEnv(sys.argv)

    if remote_env.is_remote():
        remote_env.prepare()

        rmt_work_dir = remote_env.get_work_dir()
        save_result_path = remote_env.get_save_result_path()

        pool = multiprocessing.Pool(remote_env.get_cores())
        time_start = time.time()
        # COUNT = 0
        for param in remote_env.param_generator():
            pid, zzg_num, ema_num, multiplier = param
            pool.apply_async(save_result, (
            zzg_num, ema_num, multiplier, save_result_path, rmt_work_dir + '/tools/backtesting_data/'))

        pool.close()
        pool.join()

        remote_env.mark_end_with_success()
        time_end = time.time()
        duration = time_end - time_start
        os.system("touch total_time_%f" % duration)
        # os.system("touch " + remote_env.get_batch_id())
        print("romote processing finished!")
    else:  # 直接在本地运行
        pass

# print('.' * 30, '优化开始', '.' * 30)
#  pool = multiprocessing.Pool(10)
#  i = 1
#  # [0.382,0.5,0.782,1][0.618, 0.886, 1.236,1.5,2]
#  for zzg_num in [0.382,0.5,0.782,1]:
#      for ema_num in range(13,57,2):
#          for multiplier in np.arange(0.1,2.7,0.2):
#              pool.apply_async(save_result,(zzg_num,ema_num,multiplier,save_result_path))
#              # rst.get()
#              print('i: ', i)
#              i += 1
#
#  print('.' * 30, '程序正在进行......')
#  pool.close()
#  pool.join()
#  print('.' * 30, '程序运行结束')
