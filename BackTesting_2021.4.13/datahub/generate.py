#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# K线生成
# 指标生成

# Author: Tayii
# Data : 2020/12/02
# ----------------------------------------------------
from dataclasses import dataclass, field
from importlib import import_module
import multiprocessing
from typing import Optional, Union, Dict
import numpy as np
import pandas as pd
import math
import datetime
import time
import logging

from backtester.template import BackTestingDataSetting
from dc.config import DataSourceConfig
from dc.source_data import get_source_data_from_config
from strategies.template import StrategyTemplate
from utility import catch_except
from .fmt import DataBase
from constant import Sec, Direction
from datahub.template import DataProcessTemplate


class BarGenerator(DataProcessTemplate):
    """
    拼接 由小周期生成大周期的新bar
    1. generating 1 minute bar data from tick data
    2. generating y minute bar data from x minute data

    Notice:
        y支持1,2,3,4,5,6小时的分钟数，或者1,2,3,5,10,15,30分钟
        y必须是x的整数倍
    """

    def __init__(self,
                 q_from_caller: multiprocessing.Queue,  # 接收来自调用函数数据的queue
                 q_to_caller: multiprocessing.Queue,  # 发送数据给调用者的queue
                 data_set_name: str,  # 数据配置 name
                 data_source: DataSourceConfig,  # 数据源配置
                 strategy_import_path: str,  # 策略包完整路径
                 ):
        """Constructor"""
        DataProcessTemplate.__init__(self, f'{data_set_name} Bar Generate',
                                     q_from_caller, q_to_caller)

        self.strategy_import_path: str = strategy_import_path
        self.strategy: Optional[StrategyTemplate] = None  # 策略实例
        self.ds_name: str = data_set_name  # 配置名
        self.data_source: DataSourceConfig = data_source
        self.data_set: Optional[BackTestingDataSetting] = None  # 配置

        # 原始bar数据
        self.__s_data: Optional[pd.DataFrame] = None
        self.__s_size: int = 0
        self.__s_data_doing_index: int = 0  # 原始bar数据 正在处理的位置index

        # 最终结果，长度跟原始数据一样，计算是基于新bar的每一个切片
        self.__ok_bar_slices: Optional[pd.DataFrame] = None
        self.__ok_bar_slices_waiting_send_index: int = 0  # 即将要发送的数据的指针位置
        self.__ok_status_bar_index: int = -1  # 已生成的新bar当前状态下 最新bar index
        self.__ok_status_latest_bar_t = None  # 已生成的新bar当前状态下 最新bar 开始时间

        self.__s_data_interval: int = 0  # 原始数据周期
        self.__ok_bars_interval: int = 0  # 新bar周期
        self.day_open: datetime.time = datetime.time(0, )
        self.night_open: Optional[datetime.time] = None

    @property
    def has_sent_index(self) -> int:
        return self.__ok_bar_slices_waiting_send_index - 1

    @property
    def ok_bar_slices_done_index(self):
        """最后已ok的指针位置"""
        return len(self.__ok_bar_slices) - 1 if self.__ok_bar_slices is not None else -1

    def run(self) -> None:
        """运行"""
        self.log(f'进程启动')

        # 生成策略实例
        _s = import_module(self.strategy_import_path).Strategy
        self.strategy: StrategyTemplate = _s(
            name=f'{self.name} strategy',  # 此处支持一次回测多个数据源
            data_source=self.data_source,
        )
        self.strategy.import_path = self.strategy_import_path

        self.data_set: BackTestingDataSetting = self.strategy.data_sets[self.ds_name]

        self.__s_data_interval = self.data_set.source_data_interval
        self.__ok_bars_interval = self.data_set.new_bar_interval
        self.day_open = self.data_set.new_bar_day_open
        self.night_open = self.data_set.new_bar_night_open

        self.log(f'☆ 开始加载回测用的原始数据...')

        # 读取原始数据
        df = self.__get_source_data(self.data_set)
        if df is None or len(df) == 0:
            self.log(f'__get_source_data err', logging.ERROR)
            return
        else:
            df = df[:]  # todo del
            self.__s_data = df
            self.__s_size = len(df)

        self.log(f'☆ 加载回测原始数据 共{self.__s_size}条...')

        # 决定由tick生成还是bar生成新bar
        generate = self.by_bar if self.__s_data_interval > Sec.TICK.value else self.by_tick

        while True:
            if self.need_quit:
                return  # 退出

            if not self.can_run:
                time.sleep(2)
                continue

            # 接收主控的指令
            pass

            # 生成
            if self.__s_data_doing_index < self.__s_size:
                try:
                    generate()
                except Exception as e:
                    self.log(f'generate err: {e}', logging.ERROR)
                finally:
                    self.__s_data_doing_index += 1

            # 发送
            if self.has_sent_index < self.__s_size - 1:  # 原始数据固定长
                self.bar_data_send_to_caller()
                if self.__s_data_doing_index >= self.__s_size and self.q_to_caller_is_full:
                    # 这种情况 出现在caller接收不及数据，而自己又已经处理完，所以sleep一下
                    time.sleep(0.5)
            else:
                # 处理且发送完了 退出
                self.do_when_all_already_sent()
                return

            # 辅助显示
            if self.__s_data_doing_index % 253 == 0:
                self.log(f'sent {self.has_sent_index} / '
                         f'generated {self.__s_data_doing_index} / source {self.__s_size}')

    @catch_except()
    def __get_source_data(self,
                          data_set: BackTestingDataSetting,  # 回测数据配置文件
                          ) -> Optional[pd.DataFrame]:
        """
        获取要回测的原始数据 OHLC等 及 配置参数
        Returns:
            (df, config)
            None: 错误
        """
        df = get_source_data_from_config(data_set.data_source)
        if df is None or df.index[0] != 0:  # index要0开头 升序
            self.log(f'df.index[0] != 0', logging.ERROR)
            return

        # 是否只需要某些列
        need_columns = data_set.need_columns
        if need_columns is not None:
            if not set(need_columns).issubset(df.columns):
                self.log(f"{need_columns} not in df.columns", logging.ERROR)
                return
            else:
                df = df[need_columns]

        # 加一列，偏移开盘时间后 属于哪天 ---------
        t_offset = data_set.day_open * 60 * 60

        def f(t):
            new_t = t - t_offset
            return datetime.datetime.fromtimestamp(new_t).date()

        df['m_Date'] = df['timestamp'].apply(f)

        return df

    def bar_data_send_to_caller(self) -> None:
        """
        发送已处理好的bar数据 给主控
        """
        if self.q_to_caller_is_full:
            return

        if self.has_sent_index < self.ok_bar_slices_done_index:
            # 有可以发送的数据
            self.send_to_caller(self.__ok_bar_slices_waiting_send_index,
                                self.__ok_bar_slices.iloc[self.__ok_bar_slices_waiting_send_index])
            self.__ok_bar_slices_waiting_send_index += 1

    @staticmethod
    def __get_dt_by_time(dt: datetime.datetime, days: int, time_: datetime.time):
        # 设置开始时间（供第1bar检索起始用）
        return (dt - datetime.timedelta(days=days)).replace(
            hour=time_.hour,
            minute=time_.minute,
            second=time_.second,
            microsecond=0)

    def by_bar(self):
        """
        由bars数据生成 小周期->大周期
        """

        def _find_nearest_dt(start_: datetime.datetime, end: datetime.datetime):
            """找到此bar属于的进入时间"""
            m = (end - start_).total_seconds() // self.__ok_bars_interval
            return start_ + datetime.timedelta(seconds=self.__ok_bars_interval * m)

        # 当前要处理 原始数据的bar
        s_bar = self.__s_data.iloc[self.__s_data_doing_index].copy()
        s_bar_dt: datetime.datetime = s_bar['datetime']  # 这条数据的开始时间
        _interval = datetime.timedelta(seconds=self.__ok_bars_interval)

        def _new_bar(prior_dt: datetime.datetime) -> None:
            """
            目标周期 添加新bar
            Args:
                prior_dt: 目标周期上个bar 开始的datetime
            """
            self.__ok_status_bar_index += 1
            self.__ok_status_latest_bar_t = _find_nearest_dt(prior_dt, s_bar_dt)  # 进入时间
            s_bar['datetime'] = self.__ok_status_latest_bar_t  # 修改开始时间
            s_bar['new_index'] = self.__ok_status_bar_index
            if self.__s_data_doing_index == 0:
                self.__ok_bar_slices = pd.DataFrame(s_bar).T
            else:
                self.__ok_bar_slices = self.__ok_bar_slices.append(s_bar, ignore_index=True)

        # 主逻辑 ===================
        s_day_open = self.__get_dt_by_time(s_bar_dt, 0, self.day_open)
        s_night_open = self.__get_dt_by_time(s_bar_dt, 0, self.night_open) \
            if self.night_open else None

        if self.__s_data_doing_index == 0:  # 第一个 刚开始处理
            if self.night_open and s_bar_dt.time() < self.day_open:
                # 从前一日夜盘开盘 开始
                start = self.__get_dt_by_time(s_bar_dt, 1, self.night_open)
            elif self.night_open is None or s_bar_dt.time() < self.night_open:
                start = s_day_open  # 日盘开盘 开始
            else:
                start = s_night_open  # 夜盘开盘 开始

            _new_bar(start)  # 新bar
            return

        # 如果跨过开盘时间，直接设置为新bar
        if self.__ok_status_latest_bar_t < s_day_open <= s_bar_dt:
            _new_bar(s_day_open)  # 新bar
            return
        elif self.night_open and self.__ok_status_latest_bar_t < s_night_open <= s_bar_dt:
            _new_bar(s_night_open)  # 新bar
            return

        # 如果超过周期时间
        if (s_bar_dt - self.__ok_status_latest_bar_t).total_seconds() >= self.__ok_bars_interval:
            _new_bar(self.__ok_status_latest_bar_t)  # 新bar
            return

        # 都不是，则是一个大周期内，拼接 --------------------
        ok_bar: pd.Series = self.__ok_bar_slices.iloc[self.ok_bar_slices_done_index].copy()
        for item in ok_bar.keys():
            # 必选 OHLC  open不动
            if item in ['High', 'high']:
                ok_bar[item] = max(ok_bar[item], s_bar[item])
            elif item in ['Low', 'low']:
                ok_bar[item] = min(ok_bar[item], s_bar[item])
            elif item in ['Close', 'close', 'Last', 'last']:
                ok_bar[item] = s_bar[item]
            elif item in ['Vol', 'vol', 'Volume', 'volume']:
                ok_bar[item] += s_bar[item]
            # 日期之类 new_index 不变
            # 其他 todo
        # 保存新bar切片
        self.__ok_bar_slices = self.__ok_bar_slices.append(ok_bar, ignore_index=True)

    def by_tick(self):
        """由tick数据生成
        pass

#         Update new tick data into generator.
#         """


@dataclass
class Performance(DataBase):
    """
    性能指标
    年化波动率，年化收益率，累积收益率，夏普比，最大回撤，卡玛比率，年化下行波动率，
    索提诺比率，skeness,kurtosis,average_Top5MaxDrawdown
     annualStd,sharpeRatio,maxDrawdown,calmarRatio,annualDownsideStd,sortinoRatio,skewness,
     kurtosis,averageTop5MaxDrawdown,annualReturn,cumReturn
    """
    name: str
    init_asset: float  # 初始总资产

    df_all: pd.DataFrame  # 成交结果
    df_long: pd.DataFrame = field(init=False)  # long成交结果
    df_short: pd.DataFrame = field(init=False)  # short成交结果

    days: datetime.timedelta = field(init=False)  # 总天数

    # 累计收益
    total_net_profit: dict = field(init=False)
    total_earnings_rate: dict = field(init=False)

    # 按日收益
    day_earnings: dict = field(init=False)
    day_earnings_rate: dict = field(init=False)
    day_pnl: dict = field(init=False)
    day_trades: dict = field(init=False)

    # 每日平均最大止盈止损
    day_max_floating_profit_mean: dict = field(init=False)
    day_max_floating_loss_mean: dict = field(init=False)
    day_max_floating_diff_mean: dict = field(init=False)  # 差值
    day_max_floating_profit_std: dict = field(init=False)
    day_max_floating_loss_std: dict = field(init=False)
    day_max_floating_diff_std: dict = field(init=False)

    calc_list = ['all', 'long', 'short']  # 计算用的数据 [全部 按开多 按开空]

    def __post_init__(self):
        if self.df_all is None or len(self.df_all) == 0:
            return

        if 'open_date' not in self.df_all.columns:
            self.df_all['open_date'] = self.df_all.apply(
                lambda x: x['open_datetime'].date(), axis=1)

        self.df_long = self.df_all[self.df_all['open_direction'] == 'Direction.LONG']
        self.df_long.reset_index(inplace=True)
        self.df_short = self.df_all[self.df_all['open_direction'] == 'Direction.SHORT']
        self.df_short.reset_index(inplace=True)

        self.days = (self.df_all.iloc[-1]['close_datetime'] -
                     self.df_all.iloc[0]['open_datetime'])

        # 累计收益
        self.total_net_profit, self.total_net_profit_rate = {}, {}

        # 日收益
        self.day_earnings, self.day_earnings_rate, self.day_pnl = {}, {}, {}
        self.day_trades = {}
        self.day_max_floating_profit_mean, self.day_max_floating_loss_mean = {}, {}
        self.day_max_floating_profit_std, self.day_max_floating_loss_std = {}, {}
        self.day_max_floating_diff_std, self.day_max_floating_diff_std = {}, {}

    def __repr__(self):
        return f'{self.days} {len(self.df_all)} {self.total_net_profit}'

    def calc_trades(self) -> dict:
        """计算交易单量相关数据"""
        ret: dict = {}
        for c in self.calc_list:
            df = eval(f'self.df_{c}')
            # 总的单数
            ret[f'trade_num.all.{c}'] = len(df)
            # 盈利的单数
            ret[f'trade_num.winning.{c}'] = len(df[df['earnings'] > 0])
            # 亏损的单数
            ret[f'trade_num.losing.{c}'] = ret[f'trade_num.all.{c}'] - ret[f'trade_num.winning.{c}']
            # 盈利占比
            if ret[f'trade_num.all.{c}'] != 0:
                ret[f'percent_profitable.{c}'] = ret[f'trade_num.winning.{c}'] / ret[f'trade_num.all.{c}']
        # 输出4个指标
        return ret

    def calc_earnings(self) -> dict:
        """计算盈利相关数据"""
        ret: dict = {}
        for c in self.calc_list:
            df = eval(f'self.df_{c}')

            # 总的净利润
            ret[f'total_net_profit.{c}'] = df['earnings'].sum()
            self.total_net_profit[c] = ret[f'total_net_profit.{c}']
            # 总的净利润率
            ret[f'total_net_profit_rate.{c}'] = self.total_net_profit[c] / self.init_asset
            self.total_net_profit_rate[c] = ret[f'total_net_profit_rate.{c}']

            # 平均净利润std
            ret[f'avg_net_profit.all.{c}'] = ret[f'total_net_profit.{c}'] / len(df)
            ret[f'std_net_profit.all.{c}'] = df['earnings'].std()
            # 最大浮盈 均值 std
            ret[f'avg_net.max_floating_profit.{c}'] = df['max_floating_profit'].mean()
            ret[f'std_net.max_floating_profit.{c}'] = df['max_floating_profit'].std()
            #  最大浮亏 均值 std
            ret[f'avg_net.max_floating_loss.{c}'] = df['max_floating_loss'].mean()
            ret[f'std_net.max_floating_loss.{c}'] = df['max_floating_loss'].std()
            # 最大浮盈-最大浮亏 均值 std
            df[f'max_floating_diff'] = (df[f'max_floating_profit'] +
                                             df[f'max_floating_loss'])
            ret[f'avg_net.max_floating_diff.{c}'] = df['max_floating_diff'].mean()
            ret[f'std_net.max_floating_diff.{c}'] = df['max_floating_diff'].std()

            # 赢单 平均净利润 std
            df_ = df[df['earnings'] > 0]
            ret[f'avg_net_profit.winning.{c}'] = df_['earnings'].mean()
            ret[f'std_net_profit.winning.{c}'] = df_['earnings'].std()
            # 赢单 最大浮盈 均值 std
            ret[f'avg_net.max_floating_profit.winning.{c}'] = df_['max_floating_profit'].mean()
            ret[f'std_net.max_floating_profit.winning.{c}'] = df_['max_floating_profit'].std()
            # 赢单 最大浮亏 均值 std
            ret[f'avg_net.max_floating_loss.winning.{c}'] = df_['max_floating_loss'].mean()
            ret[f'std_net.max_floating_loss.winning.{c}'] = df_['max_floating_loss'].std()

            # 亏损单 平均净亏损 std
            df_ = df[df['earnings'] <= 0]
            ret[f'avg_net_profit.losing.{c}'] = abs(df_['earnings'].mean())
            ret[f'avg_net_profit.losing.{c}'] = df_['earnings'].mean()
            # 亏损单 最大浮盈 均值 std
            ret[f'avg_net.max_floating_profit.losing.{c}'] = df_['max_floating_profit'].mean()
            ret[f'std_net.max_floating_profit.losing.{c}'] = df_['max_floating_profit'].std()
            # 亏损单 最大浮亏 均值 std
            ret[f'avg_net.max_floating_loss.losing.{c}'] = df_['max_floating_loss'].mean()
            ret[f'std_net.max_floating_loss.losing.{c}'] = df_['max_floating_loss'].std()

            # 盈亏比
            ret[f'ratio_winning_losing.{c}'] = (ret[f'avg_net_profit.winning.{c}']
                                                / ret[f'avg_net_profit.losing.{c}'])

            # ret[f'gross_profit.{c}'] =  # 毛利润

        # 输出5个指标
        return ret

    def calc_duration(self):
        """计算开仓时间相关数据"""
        ret: dict = {}
        for c in self.calc_list:
            df = eval(f'self.df_{c}')
            ret[f'avg_duration.{c}'] = df['duration'].mean()
            if len(df):
                ret[f'avg_deal.{c}'] = int(self.days.total_seconds() / len(df))
        return ret

    # def calc_annual_earnings(self):

    def calc_day_earnings(self,
                          calc_max_floating=False,
                          ) -> Dict[str, pd.Series]:
        """按日收益 收益率（按开仓日）"""
        ret: dict = {}
        for c in self.calc_list:
            df = eval(f'self.df_{c}')

            if 'open_date' not in df.columns:
                df['open_date'] = df.apply(
                    lambda x: x['open_datetime'].date(), axis=1)

            # 每日盈利
            grouped_earnings = df.groupby('open_date')['earnings']
            self.day_earnings[c] = grouped_earnings.sum()
            ret[f'day_earnings.{c}'] = self.day_earnings[c]
            # 每日开单数量
            self.day_trades[c] = grouped_earnings.count()
            ret[f'day_trades.{c}'] = self.day_trades[c]

            if calc_max_floating:
                grouped_mfp = df.groupby('open_date')['max_floating_profit']
                # 每日最大浮盈 均值
                self.day_max_floating_profit_mean = grouped_mfp.mean()
                # 每日最大浮盈 std
                self.day_max_floating_profit_std = grouped_mfp.std()
                self.day_max_floating_profit_std.fillna(0, inplace=True)

                grouped_mfl = df.groupby('open_date')['max_floating_loss']
                # 每日最大浮亏 均值
                self.day_max_floating_loss_mean = grouped_mfl.mean()
                # 每日最大浮亏 std
                self.day_max_floating_loss_std = grouped_mfl.std()
                self.day_max_floating_loss_std.fillna(0, inplace=True)

                df['max_floating_diff'] = df['max_floating_loss'] + df['max_floating_profit']
                grouped_mfd = df.groupby('open_date')['max_floating_diff']
                # 每日最大浮盈浮亏差值 均值
                self.day_max_floating_diff_mean = grouped_mfd.mean()
                # 每日最大浮盈浮亏差值 std
                self.day_max_floating_diff_std = grouped_mfd.std()
                self.day_max_floating_diff_std.fillna(0, inplace=True)

            # 日累积收益 资金曲线（按开仓日）
            self.day_pnl[c] = self.day_earnings[c].cumsum()
            ret[f'day_pnl.{c}'] = self.day_pnl[c]

            df_rate = pd.DataFrame(self.day_pnl[c])
            df_rate['day_asset'] = df_rate['earnings'].shift(1) + self.init_asset  # 每日起始资产
            df_rate.loc[df_rate.index[0], 'day_asset'] = self.init_asset
            try:
                df_rate['day_earnings_rate'] = df_rate['day_asset'].shift(-1) / df_rate['day_asset']
                df_rate['day_earnings_rate'] = df_rate['day_earnings_rate'].apply(lambda x: math.log10(x))
            except:  # 资产到负 会出错
                pass

        return ret

    # todo ↓

    def period_earnings(self) -> pd.Series:
        """按时间段计算收益"""
        # if self.df is not None:
        #     print(self.df)

    def calc_day_earnings_rate(self) -> pd.Series:
        """ 按日收益率（按平仓日） """
        for c in self.calc_list:
            df = eval(f'self.df_{c}')

        # todo  ln(si / si_1)
        if self.__day_earnings_rate is None:
            if self.__day_earnings is None:
                self.__day_earnings = self.day_earnings

            self.__day_earnings_rate = self.__day_earnings.apply(
                lambda x: x / self.init_asset)

        return self.__day_earnings_rate

    @property
    def annual_std(self) -> float:
        """ 年化波动率 """
        daily_std = self.day_earnings_rate.std()  # 得到日度波动率
        annual_std = daily_std * math.sqrt(250)  # 得到年化波动率
        return daily_std, annual_std

    @property
    def annual_earnings_rate(self) -> float:
        """
        年化收益率
        超过一年  =  ((单位净值(现在) / 单位净值(成立日)) ^ (1 / 存续年数) - 1) * 100%
        不超过一年  = 存续区间累计收益率 / 存续天数 * 360  （或 存续区间累计收益率）
        """
        return self.total_earnings_rate / self.days * 250  # 暂时都按天

    def show_result(self):
        if self.df_all is None or len(self.df_all) == 0:
            self.log(f'没有成交')
        else:
            self.log(f'\n回测总天数: {self.days}天  '
                     f'初始资产:{self.init_asset} 最终资产：{self.asset:.2f} \n'
                     f'平均开仓手数:{self.mean_lots:.0%}  '
                     f'交易次数: {self.calc_trades()} '
                     f'持仓时间 {self.calc_duration()} \n'
                     f'收益:{self.calc_earnings()} \n'
                     # f'夏普比例(无风险率：{0:.2f}):{self.sharpe_ratio(rf=0):.2f} \n'
                     )

    def sharpe_ratio(self, rf: float  # 无风险利率
                     ):
        """
        夏普比例 =（预期收益率-无风险利率）/投资组合标准差.
        [E(Rp)－Rf]/σp
        """
        return (self.annual_earnings_rate - rf) / self.annual_std  # 得到夏普比

    @property
    def mean_lots(self):
        """平均开仓手数"""
        return self.df_all['open_amount'].mean()

    def max_floating_profit(self,
                            add_open_datetime: bool = False,  # 是否加上时间
                            ) -> Union[pd.DataFrame, pd.Series]:
        """最大浮盈"""
        if add_open_datetime:
            return self.df_all[['open_datetime', 'max_floating_profit']]
        else:
            return self.df_all['max_floating_profit']

    # todo 下面
    #
    # def max_draw_down(self):
    #     """
    #     最大回撤
    #     """
    #     roll_max = self.df['close'].expanding().max()
    #     maxdraw_down = -1 * np.min(self.df['close'] / roll_max - 1)  # 计算得到最大回撤
    #     return maxdraw_down
    #
    # def calmar_Ratio(self):
    #     """
    #     Returns: Calmar ratio.
    #     """
    #     annual_return = self.annual_earnings_rate(self)
    #     maxdraw_down = self.max_draw_down(self)
    #     calmarRatio = annual_return / maxdraw_down
    #     return calmarRatio
    #
    # def annual_DownsideStd(self):
    #     """
    #     Returns:Annual downside standard deviation.
    #     """
    #     num = len(self.df[self.df['return'] < 0]['return'])  # 计算小于0的收益率个数
    #     dailyDownsideStd = math.sqrt(
    #         self.df[self.df['return'] < 0]['return'].apply(lambda x: x * x).sum() / num)  # 计算出日度下行波动率
    #     annualDownsideStd = dailyDownsideStd * math.sqrt(250)
    #     return annualDownsideStd
    #
    # def sortino_Ratio(self):
    #     """
    #     Returns: Sortino ratio.
    #     """
    #     annualReturn = self.annual_earnings_rate(self)
    #     annualDownsideStd = self.annual_DownsideStd(self)
    #     sortinoRatio = annualReturn / annualDownsideStd
    #     return sortinoRatio
    #
    # def skewness(self):
    #     """
    #     Returns: The skewness of the return.
    #     """
    #     return self.df['return'].skew()
    #
    # def kurtosis(self):
    #     """
    #     Returns: The kurtosis of the return.
    #     """
    #     return self.df['return'].kurt()
    #
    # def average_Top5Maxdraw_down(self):
    #     """
    #     Returns: The average top 5 max draw_down.
    #     """
    #     draw_downList = []  # 定义一个序列，存储不同排名的最大回撤
    #     for i in range(5):
    #         # 计算最大回撤
    #         roll_max = self.df['close'].expanding().max()
    #         draw_down = -1 * np.min(self.df['close'] / roll_max - 1)  # 计算得到当前阶段最大回撤
    #         if draw_down <= 0:
    #             break
    #         draw_downList.append(draw_down)
    #
    #         # 找到最大回撤对应的起始index和终止index
    #         end_point = np.argmin(self.df['close'] / roll_max - 1)
    #         start_point = np.argmax(self.df['close'][:end_point])
    #
    #         # 将最大回撤阶段的数据去掉，将两端数据拼接，这里需要处理使得拼接点一致
    #         df1 = self.df[['close']][:start_point]
    #         df2 = self.df[['close']][end_point:]
    #         if not df1.empty and not df2.empty:
    #             df2['close'] = df2['close'] * (
    #                     df1.loc[df1.index[-1], 'close'] / df2.loc[df2.index[0], 'close'])  # 将df2的第一个数据与df1的最后一个数据一致
    #             df = pd.concat([df1, df2])  # 将df1与df2拼接，得到新的df数据
    #         elif df1.empty and not df2.empty:
    #             df = df2
    #         elif not df1.empty and df2.empty:
    #             df = df1
    #         elif df1.empty and df2.empty:
    #             break
    #     averageTop5Maxdraw_down = pd.Series(draw_downList).mean()
    #     return averageTop5Maxdraw_down

    #
    # new_minute = False
    #
    # # Filter tick data with 0 last price
    # if not tick.last_price:
    #     return
    #
    # # Filter tick data with less intraday trading volume (i.e. older timestamp)
    # if self.last_tick and tick.volume and tick.volume < self.last_tick.volume:
    #     return
    #
    # if not self.bar:
    #     new_minute = True
    # elif (self.bar.datetime.minute != tick.datetime.minute) or (self.bar.datetime.hour != tick.datetime.hour):
    #     self.bar.datetime = self.bar.datetime.replace(
    #         second=0, microsecond=0
    #     )
    #     self.on_bar(self.bar)
    #
    #     new_minute = True
    #
    # if new_minute:
    #     self.bar = BarData(
    #         symbol=tick.symbol,
    #         exchange=tick.exchange,
    #         interval=Interval.MINUTE,
    #         datetime=tick.datetime,
    #         gateway_name=tick.gateway_name,
    #         open_price=tick.last_price,
    #         high_price=tick.last_price,
    #         low_price=tick.last_price,
    #         close_price=tick.last_price,
    #         open_interest=tick.open_interest
    #     )
    # else:
    #     self.bar.high_price = max(self.bar.high_price, tick.last_price)
    #     self.bar.low_price = min(self.bar.low_price, tick.last_price)
    #     self.bar.close_price = tick.last_price
    #     self.bar.open_interest = tick.open_interest
    #     self.bar.datetime = tick.datetime
    #
    # if self.last_tick:
    #     volume_change = tick.volume - self.last_tick.volume
    #     self.bar.volume += max(volume_change, 0)
    #
    # self.last_tick = tick

# class ArrayManager(object):
#
#     def apo(
#             self,
#             fast_period: int,
#             slow_period: int,
#             matype: int = 0,
#             array: bool = False
#     ) -> Union[float, np.ndarray]:
#         """
#         APO.
#         """
#         result = talib.APO(self.close, fast_period, slow_period, matype)
#         if array:
#             return result
#         return result[-1]
#
#
#     def ppo(
#             self,
#             fast_period: int,
#             slow_period: int,
#             matype: int = 0,
#             array: bool = False
#     ) -> Union[float, np.ndarray]:
#         """
#         PPO.
#         """
#         result = talib.PPO(self.close, fast_period, slow_period, matype)
#         if array:
#             return result
#         return result[-1]
#
#
#
#     def macd(
#             self,
#             fast_period: int,
#             slow_period: int,
#             signal_period: int,
#             array: bool = False
#     ) -> Union[
#         Tuple[np.ndarray, np.ndarray, np.ndarray],
#         Tuple[float, float, float]
#     ]:
#         """
#         MACD.
#         """
#         macd, signal, hist = talib.MACD(
#             self.close, fast_period, slow_period, signal_period
#         )
#         if array:
#             return macd, signal, hist
#         return macd[-1], signal[-1], hist[-1]
#
#
#
#     def ultosc(
#             self,
#             time_period1: int = 7,
#             time_period2: int = 14,
#             time_period3: int = 28,
#             array: bool = False
#     ) -> Union[float, np.ndarray]:
#         """
#         Ultimate Oscillator.
#         """
#         result = talib.ULTOSC(self.high, self.low, self.close, time_period1, time_period2, time_period3)
#         if array:
#             return result
#         return result[-1]
#

#
#     def keltner(
#             self,
#             n: int,
#             dev: float,
#             array: bool = False
#     ) -> Union[
#         Tuple[np.ndarray, np.ndarray],
#         Tuple[float, float]
#     ]:
#         """
#         Keltner Channel.
#         """
#         mid = self.sma(n, array)
#         atr = self.atr(n, array)
#
#         up = mid + atr * dev
#         down = mid - atr * dev
#
#         return up, down
#
#     def donchian(
#             self, n: int, array: bool = False
#     ) -> Union[
#         Tuple[np.ndarray, np.ndarray],
#         Tuple[float, float]
#     ]:
#         """
#         Donchian Channel.
#         """
#         up = talib.MAX(self.high, n)
#         down = talib.MIN(self.low, n)
#
#         if array:
#             return up, down
#         return up[-1], down[-1]
#
#     def aroon(
#             self,
#             n: int,
#             array: bool = False
#     ) -> Union[
#         Tuple[np.ndarray, np.ndarray],
#         Tuple[float, float]
#     ]:
#         """
#         Aroon indicator.
#         """
#         aroon_up, aroon_down = talib.AROON(self.high, self.low, n)
#
#         if array:
#             return aroon_up, aroon_down
#         return aroon_up[-1], aroon_down[-1]
#
#     def aroonosc(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
#         """
#         Aroon Oscillator.
#         """
#         result = talib.AROONOSC(self.high, self.low, n)
#
#         if array:
#             return result
#         return result[-1]
#
#
#     def mfi(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
#         """
#         Money Flow Index.
#         """
#         result = talib.MFI(self.high, self.low, self.close, self.volume, n)
#         if array:
#             return result
#         return result[-1]
#
#     def ad(self, array: bool = False) -> Union[float, np.ndarray]:
#         """
#         AD.
#         """
#         result = talib.AD(self.high, self.low, self.close, self.volume)
#         if array:
#             return result
#         return result[-1]
#
#     def adosc(
#             self,
#             fast_period: int,
#             slow_period: int,
#             array: bool = False
#     ) -> Union[float, np.ndarray]:
#         """
#         ADOSC.
#         """
#         result = talib.ADOSC(self.high, self.low, self.close, self.volume, fast_period, slow_period)
#         if array:
#             return result
#         return result[-1]
#
#     def bop(self, array: bool = False) -> Union[float, np.ndarray]:
#         """
#         BOP.
#         """
#         result = talib.BOP(self.open, self.high, self.low, self.close)
#
#         if array:
#             return result
#         return result[-1]
