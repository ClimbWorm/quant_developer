import numpy as np
import pandas as pd
import multiprocessing
import talib
from dataclasses import dataclass, field
from strategy.define_enum_constant import *
from strategy.functionsforATRRegression import *
from strategy.order_management import *
from tools.backtesting_backup.get_specific_enter_price import *
from tools.indicators.Indicators import AddRSIDivergence, AddRSI_T, RSI_SMA_shortgo

epsilon = 1e-5


@dataclass
class Strategy(object):

    def __init__(self, zzg_num):
        # path = "F:\\pythonHistoricalTesting\\pythonHistoricalTesting\\code_generated_csv\\ema21_trailing_stop\\"
        # filename = "df_backtesting_all_rsi_ohlc_avg_{}.csv".format(zzg_num)
        # self.df_backtesting: pd.DataFrame = pd.read_csv(path+filename, index_col=0)
        # self.df_backtesting: pd.DataFrame = self.process_df_backtesting()  # 回测数据
        self.df_backtesting: pd.DataFrame = pd.DataFrame()
        self.zzg_num = zzg_num
        self.is_alive_orders = []  # list中为一个个order的实例
        # 记录历史记录
        self.his_table = pd.DataFrame(columns=['entry_idx', 'EntryTime'])
        self.df_all_zzg = self.generate_zzg_table_include_once_and_final()
        #
        # # 与zzp相关的
        # self.last_zzg_bar_Idx, self.last_extreme_bar_low_or_high, self.last_extreme_value, self.last_extreme_bar_threshold \
        #     = lastExtremeBar(0, self.df_all_zzg)

    def add_ema(self, df_backtesting, period=21):
        df_backtesting[f'ema_{period}'] = talib.EMA(df_backtesting.Close,
                                                    timeperiod=period)  # .shift(1) ema的计算是包括当前bar的
        return df_backtesting

    def add_rsi(self, df_backtesting, df_60min, period=60, input_data=1):
        """input_data：1 表示close 2 表示用开高低收的均值
        最后一个close或者开高低收的均值取当前的15min的bar，之前的数据都取60min的"""
        rsi_list = []
        for t in range(len(df_backtesting)):
            time = pd.to_datetime(df_backtesting.Time.loc[t])
            time_60min = time.strftime("%Y-%m-%d %H:00:00")
            t_ = df_60min.loc[df_60min.Time == time_60min].index[0]  # 60min表中的index
            if time.strftime("%M") == "45":
                # 包括当前bar，只要在df_60min中取14个数即可
                if input_data == 1:
                    try:
                        rsi_sub = talib.RSI(df_60min.Close.loc[t_ - 13:t_], timeperiod=13)[t_]
                    except Exception as e:
                        rsi_sub = np.nan
                else:
                    try:
                        avg_OHLC = np.mean(
                            [df_60min.Open.loc[t_ - 13:t_], df_60min.High.loc[t_ - 13:t_], df_60min.Low.loc[t_ - 13:t_],
                             df_60min.Close.loc[t_ - 13:t_]], axis=0)
                        rsi_sub = talib.RSI(pd.Series(avg_OHLC),timeperiod=13)[13]
                    except Exception as e:
                        rsi_sub = np.nan
            else:
                # 当前bar的数值（在df_backtesting这个15min的表中找，再往前取13个数）
                if input_data == 1:
                    try:
                        data_series = df_60min.Close.loc[t_ - 13:t_ - 1].tolist() + [df_backtesting.Close.loc[t]]
                        rsi_sub = talib.RSI(pd.Series(data_series),timeperiod=13)[13]
                    except Exception as e:
                        rsi_sub = np.nan
                else:
                    try:
                        avg_OHLC = np.mean([df_60min.Open.loc[t_ - 13:t_ - 1], df_60min.High.loc[t_ - 13:t_ - 1],
                                            df_60min.Low.loc[t_ - 13:t_ - 1], df_60min.Close.loc[t_ - 13:t_ - 1]],
                                           axis=0)
                        data_series = avg_OHLC.tolist() + [np.mean(
                            [df_backtesting.Open.loc[t], df_backtesting.High.loc[t], df_backtesting.Low.loc[t],
                             df_backtesting.Close.loc[t]])]
                        rsi_sub = talib.RSI(pd.Series(data_series),timeperiod=13)[13]
                    except Exception as e:
                        rsi_sub = np.nan
            rsi_list.append(rsi_sub)
        df_backtesting[f'rsi_{period}_min'] = rsi_list
        return df_backtesting

    # 专门用来生成数据，只在程序main运行之前运行一次，提前储存好数据
    def process_df_backtesting(self):
        df_backtesting = pd.read_csv(
            'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/back.csv')
        df_60min = pd.read_csv(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\backtesting\BackTesting\back_testing_result\ema21_trailing_stop_ha_rsi\start_from_2015_YM_min60.csv',
            index_col=0)
        # 对60min的表做去重处理，只保留每一个小时的最后一条记录
        df_60min = df_60min.loc[:, ["datetime", "Open", "High", "Low", "Last"]]
        df_60min.columns = ["Time", "Open", "High", "Low", "Close"]
        df_60min = df_60min.drop_duplicates(subset=['Time'], keep='last').reset_index(drop=True)
        df_backtesting['Time'] = pd.to_datetime(df_backtesting['Time'])
        df_backtesting = get_atr_breakout_price(df_backtesting, 1.618)
        for i in range(13,56):
            df_backtesting = self.add_ema(df_backtesting,period=i)
        # 添加60min的rsi
        df_backtesting = self.add_rsi(df_backtesting, df_60min, input_data=2)
        # print("运行完add_rsi",df_backtesting)
        # 添加rsi背离
        df_backtesting['RSI'] = talib.RSI(df_backtesting.Close, timeperiod=13)
        df_backtesting = AddRSIDivergence(df_backtesting, self.df_all_zzg)
        # print("运行完rsi背离", df_backtesting)
        # 添加rsi上下穿
        df_backtesting = AddRSI_T(df_backtesting, 2)
        df_backtesting = AddRSI_T(df_backtesting, 7)
        df_backtesting = RSI_SMA_shortgo(df_backtesting, 3)
        # print("运行完rsi上下穿", df_backtesting)
        # 做数据清洗
        df_backtesting = df_backtesting[
            ~df_backtesting.isin([np.nan, np.inf, -np.inf]).any(1)]  # 这边不要reset_index，后面注意loc和iloc的使用
        # print("运行完数据清洗", df_backtesting)
        # df_backtesting.to_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\cleaned_data.csv')
        df_backtesting.to_csv(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\df_backtesting_all_rsi_ohlc_avg_{}.csv'.format(self.zzg_num))
        return df_backtesting

    def generate_zzg_table_include_once_and_final(self):
        zzg_num = self.zzg_num
        df_zigzag = pd.read_csv(
            "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210331_{}.csv".format(
                zzg_num),
            index_col=0)
        df_all_zzg = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
                df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
        df_all_zzg["bar_num"] = df_all_zzg.index
        df_all_zzg = df_all_zzg.reset_index(drop=True)
        df_all_zzg['m_Date'] = pd.to_datetime(df_all_zzg['m_Date'], format="%Y-%m-%d")
        return df_all_zzg

    def judge_open_direction(self):
        if (self.last_extreme_bar_low_or_high == "ZZPT.ONCE_HIGH") | (
                self.last_extreme_bar_low_or_high == "ZZPT.HIGH"):
            return TradeDirection.SHORT
        elif (self.last_extreme_bar_low_or_high == "ZZPT.ONCE_LOW") | (
                self.last_extreme_bar_low_or_high == "ZZPT.LOW"):
            return TradeDirection.LONG
        else:
            pass

    def can_be_optimized_entry(self, i_) -> bool:
        if self.df_backtesting.loc[i_].is_atr_breakout:
            return True
        else:
            return False

    def can_open(self, i_, open_direction_) -> bool:
        if self.can_be_optimized_entry(i_):
            if open_direction_ == TradeDirection.SHORT:
                if (self.df_backtesting.loc[i_].HA_Close - self.df_backtesting.loc[i_].HA_Open) < 0:
                    if (80 < self.df_backtesting.loc[i_].rsi_60_min < 100)|(self.df_backtesting.loc[i_].RSIShortgodown == 1)|(self.df_backtesting.loc[i_].RSI_TopDivergence == 1):
                        return True
                return False
            else:
                if (self.df_backtesting.loc[i_].HA_Close - self.df_backtesting.loc[i_].HA_Open) > 0:
                    if (0 < self.df_backtesting.loc[i_].rsi_60_min < 20)|(self.df_backtesting.loc[i_].RSIShortgoup == 1)|(self.df_backtesting.loc[i_].RSI_BottomDivergence == 1):
                        return True
                return False
        else:  # 没有突破
            if open_direction_ == TradeDirection.SHORT:
                if (self.df_backtesting.loc[i_ - 1].HA_Close - self.df_backtesting.loc[i_ - 1].HA_Open) < 0:
                    if (80 < self.df_backtesting.loc[i_-1].rsi_60_min < 100)|(self.df_backtesting.loc[i_-1].RSIShortgodown == 1)|(self.df_backtesting.loc[i_-1].RSI_TopDivergence == 1):
                        return True
                return False
            else:
                if (self.df_backtesting.loc[i_ - 1].HA_Close - self.df_backtesting.loc[i_ - 1].HA_Open) > 0:
                    if (0 < self.df_backtesting.loc[i_-1].rsi_60_min < 20)|(self.df_backtesting.loc[i_-1].RSIShortgoup == 1)|(self.df_backtesting.loc[i_-1].RSI_BottomDivergence == 1):
                        return True
                return False

    def is_limit_open_by_same_previous_zzp(self, last_zzg_bar_Idx_, is_alive_orders_, open_direction_) -> bool:
        if len(is_alive_orders_) == 0:
            return True
        for order_ in is_alive_orders_[-30:]:  # Todo 这边对于不需要获取离场信息的，就只遍历最后加入的30个即可
            if order_.last_zzp_index == last_zzg_bar_Idx_:
                if order_.open_direction == open_direction_:
                    return False
        return True

    def new_an_order_object(self, entry_idx_, EntryTime_):
        _new_trade = Orders(entry_idx=entry_idx_,
                            EntryTime=EntryTime_, )  # 实例化一个订单对象

        self.is_alive_orders.append(_new_trade)
        # Todo 对于不需要获取最终离场信息的，还要打开下面这行
        self.his_table = self.his_table.append(_new_trade.result_as_series(), ignore_index=True)

    def main(self):
        """如果有同一个前zzp的同向订单限制再开同向订单，没有反向信号出现就平仓的逻辑"""
        traverse_list = set(self.df_backtesting.index) & set(self.df_backtesting.index + 1)
        traverse_list = sorted(traverse_list)
        for i in traverse_list:  # range(3000,e):# 因为前面有些信息缺失2000
            # CurrentTime = self.df_backtesting.loc[i].Time # Todo 使用事先生成好的数据这里有一个弊端，就是原本时间格式的时间，读入又变成str格式了
            CurrentTime = pd.to_datetime(self.df_backtesting.loc[i].Time)
            # PreviousTime = self.df_backtesting.loc[i - 1].Time
            # current_date = pd.to_datetime(CurrentTime.strftime("%Y-%m-%d"))

            # 计算当前bar距离上一个极值点的距离,获取到上一个极值点的类型，是高点还是低点及其数值
            self.last_zzg_bar_Idx, self.last_extreme_bar_low_or_high, self.last_extreme_value, self.last_extreme_bar_threshold = \
                lastExtremeBar(i, self.df_all_zzg)

            if self.last_zzg_bar_Idx == 0:
                continue

            distance = i - self.last_zzg_bar_Idx
            if ((self.zzg_num < 0.886) & (distance > 5)) | ((self.zzg_num >= 0.886) & (distance > 10)):
                continue
            else:
                open_direction = self.judge_open_direction()
                # 进场
                if self.can_open(i, open_direction):
                    if self.is_limit_open_by_same_previous_zzp(self.last_zzg_bar_Idx, self.is_alive_orders,
                                                               open_direction):
                        self.new_an_order_object(entry_idx_=i, EntryTime_=CurrentTime, )
        return self.his_table


def save_result(zzg_num, open_threshold_multiplier):
    import matplotlib.pyplot as plt
    strategy = Strategy(zzg_num)
    strategy.main()
    whole_result = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\final_version_only_ha_ema21_trailing_stop\table_{}_{}.csv'.format(
            zzg_num, open_threshold_multiplier), index_col=0)
    common_open_idx = set(whole_result.entry_idx) & set(strategy.his_table.entry_idx)
    filtered_table = whole_result.loc[whole_result.entry_idx.isin(common_open_idx)].reset_index(drop=True)
    plt.figure(figsize=(20, 8))
    plt.plot(filtered_table.net_profit.cumsum().values)
    plt.title('pnl_add_rsi')
    plt.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\ha_rsi_all_added_ohlc_avg\pnl_add_rsi_{}_{}.png'.format(
            zzg_num, open_threshold_multiplier))
    filtered_table.to_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\ha_rsi_all_added_ohlc_avg\table_add_rsi_{}_{}.csv'.format(
            zzg_num, open_threshold_multiplier))

def generate_rsi_all_df(zzg_num):
    strategy = Strategy(zzg_num)
    rst = strategy.process_df_backtesting()
    print(rst)


if __name__ == '__main__':
    # 单个测试的代码
    # save_result(1.236,1)
    # strategy = Strategy(1.236)
    # strategy.main()
    # strategy.his_table.to_csv(
    #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\ema21_trailing_stop\table_{}_{}.csv'.format(
    #         1.236,1))
    # generate_rsi_all_df(2)

    generate_rsi_all_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
    pool = multiprocessing.Pool(9)
    for num in generate_rsi_all_list:
        pool.apply_async(generate_rsi_all_df,(num,))
    pool.close()
    pool.join()




    # arg_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1]
    # multiplier_list = [1.236, 1.5, 2]
    # # multiplier_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
    # print('.' * 30, '优化开始', '.' * 30)
    # pool = multiprocessing.Pool(9)
    # i = 1
    # for num in arg_list:
    #     for multiplier in multiplier_list:
    #         pool.apply_async(save_result, (num, multiplier,))
    #         print('i: ', i)
    #         i += 1
    # print('.' * 30, '程序正在进行......')
    # pool.close()
    # pool.join()
    # print('.' * 30, '程序运行结束')
