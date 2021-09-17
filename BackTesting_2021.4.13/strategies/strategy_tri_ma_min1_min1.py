#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略 3均线  只基于1分钟线 多周期调
# 进程 出场都用1分钟

# Author: Tayii
# Data : 2021/03/05
# ----------------------------------------------------
# from typing import Any, Callable#
# from datahub import BarGenerator, ArrayManager, TickData, BarData, OrderData, TradeData, StopOrder
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union
import datetime
import pandas as pd
import logging
from os import path
import numpy

from config import BACK_TESTING_RESULT_DIR
from datahub.fmt import SimuTrade
from constant import Direction, OpenMax
from datahub.indicator import IndicatorCalc as Ind
from dc.config import DataSourceConfig
from utility import catch_except, timeit
from strategies.template import StrategyTemplate, TradeParams, OpenCloseParams
from backtester.template import BackTestingDataSetting
from constant import Sec


def set_bt_data_setting(data_source: DataSourceConfig,  # 数据源配置
                        new_bar_interval: int,  # 新bar 周期
                        indicators: Dict,  # 计算的指标
                        ) -> BackTestingDataSetting:
    """设置 多周期dataset继承类"""
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


bolling_line_list = range(10, 31, 5)  # 布林带length选择范围
min1_mp_list = [21, 34, 55, 89, 144, 233, 377, 610]  # 均值周期选择范围
# min5_bl_dev = numpy.arange(0.5, 2.1, 0.25)  # 布林带 偏差倍数
min1_ma_comb = [(a, b, c) for a in min1_mp_list for b in min1_mp_list
                for c in min1_mp_list if a < b < c]  # 均值周期选择范围组合

min1_indicators = {}  # 需要预处理的数据指标
# 加 批量指标
for i in min1_mp_list:
    min1_indicators[f'ema_{i}'] = eval(f'lambda df: Ind.ema(df, {i})')
for j in bolling_line_list:
    min1_indicators[f'bolling_sma_{j}'] = eval(f'lambda df: Ind.boll(df, n={j})')

min5_indicators = {}  # 需要预处理的数据指标
# 加 批量指标
for j in bolling_line_list:
    min5_indicators[f'bolling_ema_{j}'] = eval(f'lambda df: Ind.boll(df, n={j}, ma_type="ema")')

# 超参数
hyper_parameter = [{"min1_ma_periods": min1_ma_periods,  # 一分钟线 4个周期
                    'min1_ma_type': min1_ma_type,
                    'min1_bolling_len': min1_bolling_len,
                    'min1_in_long_bl_dev': min1_in_long_bl_dev,  # 进场 1分钟布林带 dev -for long
                    'min1_out_long_bl_dev': min1_out_long_bl_dev,  # 出场 1分钟布林带 dev -for long
                    'min1_bolling_ma_type': min1_bolling_ma_type,
                    # 进场
                    'decide_open_price_type': decide_open_price_type,  # 开仓判断时用什么的价格
                    'decide_open_ma_patten': decide_open_ma_patten,  # 开仓判断时 均线图形正相关还是负相关
                    # 出场
                    'take_profit_type': take_profit_type,  # 止盈方式
                    'bl_out_price_type': bl_out_price_type,  # 布林带出场判断时用什么的价格
                    'close_types': close_types,  # 触发平仓类型
                    "stop_loss_type": stop_loss_type,
                    'fixed_out_price_type': fixed_out_price_type,  # 固定止损出场判断时用什么的价格
                    'stop_loss_fix_p': stop_loss_fix_p,  # 固定止损比例
                    }
                   #
                   for min1_ma_periods in [(55, 233, 610), ]  # min1_ma_comb
                   for min1_ma_type in ['ema', ]
                   for min1_bolling_len in bolling_line_list
                   for min1_in_long_bl_dev in numpy.arange(-2.2, -0.4, 0.15)  # 下轨进场
                   for min1_out_long_bl_dev in numpy.arange(0.5, 2.3, 0.15)  # 上轨出场
                   for min1_bolling_ma_type in ['ema', 'sma'][1:]
                   # for min5_bolling_ma_type in ['ema', ]
                   # 进场
                   for decide_open_price_type in ['extreme', 'Close'][1:]  # 进场价格选择
                   for decide_open_ma_patten in ['positive_correlation', 'negative_correlation'][1:]
                   # 出场
                   for take_profit_type in ['boll_Min1', ]
                   for bl_out_price_type in ['extreme', 'Close'][:1]  # 布林带出场判断时用什么的价格
                   for close_types in [['opposite_direction', 'day_end'][:1], ]
                   for stop_loss_type in ['fixed', 'boll_II', 'boll_I', 'bar'][:1]
                   for fixed_out_price_type in ['extreme', 'Close'][1:]  # 固定止损出场判断时用什么的价格
                   for stop_loss_fix_p in numpy.arange(0.025, 0.5, 0.025)  # 固定止损比例
                   ]


#
# def select_hyper_parameter(plan_name: str) -> list:
#     """选中的部分超参数 一般用作二次回测"""
#     columns = list(hyper_parameter[0].keys())
#     total_csv_filepath = path.join(BACK_TESTING_RESULT_DIR,
#                                    f'{plan_name}', 'total_result.csv')
#     df = pd.read_csv(total_csv_filepath)
#     df = df[df['earnings'] > 0]
#     df = df[columns]
#     source: dict = df.to_dict('records')
#     ret = []
#     # 对原始数据进行处理
#     for s in source:
#         s['min1_ma_periods'] = eval(s['min1_ma_periods'])
#         s['bl_for_long_short'] = eval(s['bl_for_long_short'])
#
#     return ret


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
            need_bars_once=1,  # 一次切片回测需要最近1个bar数据

            # 回测用数据配置文件们
            data_sets={
                # 合成1分钟的数据及指标的配置
                'min1': set_bt_data_setting(
                    data_source=data_source,
                    new_bar_interval=Sec.MIN1.value,  # 新bar 周期
                    indicators=min1_indicators,
                ),
                # 合成5分钟的数据及指标的配置
                'min5': set_bt_data_setting(
                    data_source=data_source,
                    new_bar_interval=Sec.MIN5.value,  # 新bar 周期
                    indicators=min5_indicators,
                ),
            },  # self.data_sets

            # 回测需要的超参数 list
            hyper_parameter=hyper_parameter[:],
            # 交易参数 开平仓等
            trade_params=TradeParams(
                symbol=data_source.symbol[:2],
                init_money=10000,  # 初始金额 USD
                max_lots=10,
                once_lots=1,
                open_max=OpenMax.LONG_n_OR_SHORT_n,  # 开仓无限制
                # fee_rate= 0.1,  # 交易手续费 %（可选，优先于fee_amount）
                fee_amount=5.0,  # 交易手续费 固定量（可选）
                slippage=2,  # 交易滑点  tick数
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

        # 切片数据  curr当前bar
        min1, min5 = used_data['min1'].iloc[0], used_data['min5'].iloc[0]

        # 回测超参数 阈值等 -----------------------------
        min1_ma_periods = params.get('min1_ma_periods', )
        min1_ma_type = params.get('min1_ma_type')
        min1_bolling_len: int = params.get('min1_bolling_len')
        min1_in_long_bl_dev: float = params.get('min1_in_long_bl_dev')  # 进场 1分钟布林带 dev -for long
        min1_out_long_bl_dev: float = params.get('min1_out_long_bl_dev')  # 出场 1分钟布林带 dev -for long
        min1_bolling_ma_type = params.get('min1_bolling_ma_type')
        # 进场
        decide_open_price_type = params.get('decide_open_price_type', )  # 开仓判断时用什么的价格
        decide_open_ma_patten = params.get('decide_open_ma_patten', )
        # 出场
        take_profit_type = params.get('take_profit_type', )  # 止盈方式
        bl_out_price_type = params.get('bl_out_price_type', )  # 布林带出场判断时用什么的价格
        close_types = params.get('close_types', )  # 平仓类型
        stop_loss_type = params.get('stop_loss_type', )  # 止损判断时用什么的价格
        fixed_out_price_type = params.get('fixed_out_price_type', )  # 固定止损出场判断时用什么的价格
        stop_loss_fix_p = params.get('stop_loss_fix_p', )  # 固定止损比例

        # 交易参数
        trade_p: TradeParams = self.trade_params  # 交易参数

        # 参数本身取值范围
        ma_pt = ('I', 'II', 'III') if len(min1_ma_periods) == 3 else ('I', 'II', 'III', 'IV')

        # 具体数据
        try:
            min1_ma = {}  # 1分钟 MA的数值
            for t, p in zip(ma_pt, min1_ma_periods):
                min1_ma[t] = min1[f'{min1_ma_type}_{p}']

            min1_bolling: Dict[str, float] = {
                'mid': min1[f'boll_mid_{min1_bolling_ma_type}_{min1_bolling_len}'],
                'std': min1[f'boll_std_{min1_bolling_ma_type}_{min1_bolling_len}'],
            }
            min1_bolling['in_long'] = min1_bolling['mid'] + min1_in_long_bl_dev * min1_bolling['std']
            min1_bolling['in_short'] = min1_bolling['mid'] - min1_in_long_bl_dev * min1_bolling['std']
            min1_bolling['out_long'] = min1_bolling['mid'] + min1_out_long_bl_dev * min1_bolling['std']
            min1_bolling['out_short'] = min1_bolling['mid'] - min1_out_long_bl_dev * min1_bolling['std']

            # min5_bolling: dict = {}
            # min5_bolling['mid'] = min5[f'boll_mid_{min5_bolling_ma_type}_{min5_bolling_len}']
            # min5_bolling['std'] = min5[f'boll_std_{min5_bolling_ma_type}_{min5_bolling_len}']
            # min5_bolling['tp_up'] = min5_bolling['mid'] + min5_tp_bolling_dev * min5_bolling['std']
            # min5_bolling['tp_down'] = min5_bolling['mid'] - min5_tp_bolling_dev * min5_bolling['std']
            # min5_bolling['sl_up'] = min5_bolling['mid'] + min5_sl_bolling_dev * min5_bolling['std']
            # min5_bolling['sl_down'] = min5_bolling['mid'] - min5_sl_bolling_dev * min5_bolling['std']

        except Exception as e:
            self.log(f'data err: {e}')

        # 当前处理的bar的index
        curr_index = used_data['min1'].index[-1]

        # self.log(f'curr_index = {curr_index}  {used_data["min1"]["datetime"]}')

        # 具体逻辑 =============================

        def ma_patten_lone(minute_x_ma: dict) -> bool:
            """多均线 形态 是否多头排列"""
            if len(min1_ma_periods) == 3:
                return (minute_x_ma['II'] > minute_x_ma['III']
                        and minute_x_ma['I'] > minute_x_ma['III'])
            else:
                return (minute_x_ma['III'] > minute_x_ma['IV']
                        and minute_x_ma['II'] > minute_x_ma['IV']
                        and minute_x_ma['I'] > minute_x_ma['IV'])

        def ma_patten_short(minute_x_ma: dict) -> bool:
            """多均线 形态 是否空头排列"""
            if len(min1_ma_periods) == 3:
                return (minute_x_ma['II'] < minute_x_ma['III']
                        and minute_x_ma['I'] < minute_x_ma['III'])
            else:
                return (minute_x_ma['III'] < minute_x_ma['IV']
                        and minute_x_ma['II'] < minute_x_ma['IV']
                        and minute_x_ma['I'] < minute_x_ma['IV'])

        def can_open_long() -> OpenCloseParams:
            """开多判断"""
            # 短期均线MA > 最长期的
            if decide_open_ma_patten == 'positive_correlation':
                if not ma_patten_lone(min1_ma):
                    return OpenCloseParams()
            elif decide_open_ma_patten == 'negative_correlation':
                if not ma_patten_short(min1_ma):
                    return OpenCloseParams()

            # 1min 是阳线
            if min1.Last <= min1.Open:
                return OpenCloseParams()

            # open在轨道下方，且布林带实时价格向上破 in_long的布林带
            used_price_ = (min1.Last if decide_open_price_type == 'Close' else min1.High)
            if used_price_ > min1_bolling['in_long'] > min1.Open:  # open这里约等同前bar收盘价
                return OpenCloseParams(
                    enable=True, type='cross up min1_bolling',
                    price=min1.Last if decide_open_price_type == 'Close' else min1_bolling['in_long'], )
            else:
                return OpenCloseParams()

        def can_open_short() -> OpenCloseParams:
            """开空判断"""
            # 短期均线MA < 最长期的
            if decide_open_ma_patten == 'positive_correlation':
                if not ma_patten_short(min1_ma):
                    return OpenCloseParams()
            elif decide_open_ma_patten == 'negative_correlation':
                if not ma_patten_lone(min1_ma):
                    return OpenCloseParams()

            # 1min 是阴线
            if min1.Last >= min1.Open:
                return OpenCloseParams()
            # open在轨道上方，且布林带实时价格向下破 in_short布林带
            used_price_ = (min1.Last if decide_open_price_type == 'Close' else min1.Low)
            if used_price_ < min1_bolling['in_short'] < min1.Open:  # open这里约等同前bar收盘价
                return OpenCloseParams(
                    enable=True, type='cross down min1_bolling',
                    price=min1.Last if decide_open_price_type == 'Close' else min1_bolling['in_short'], )
            else:
                return OpenCloseParams()

        # 开仓逻辑开始 ======================================
        after_open_trades, result = [], []

        # 获取开仓信号 开仓信号可以统一获取）
        open_direction = Direction.NONE  # 开仓方向
        open_ret = can_open_long()
        if open_ret.enable:
            open_direction = Direction.LONG  # 开多
        else:
            open_ret = can_open_short()
            if open_ret.enable:
                open_direction = Direction.SHORT  # 开空

        # 开仓处理
        def _process_open(trade_: SimuTrade):
            try:
                # 没开仓 且 满足开仓 用SimTrade.set_open
                trade_.set_open(open_direction,
                                curr_index,  # 当前bar
                                open_ret.price,  # 当前价
                                amount=trade_p.once_lots,  # 开仓量
                                datetime_=min1['datetime'],  # 开仓时间
                                stop_loss=(min1['Low']
                                           if open_direction == Direction.LONG
                                           else min1['High'])
                                )
            except Exception as e:
                self.log(f'work err: {e}', logging.ERROR, exc_info=True)
            finally:
                # 返回全部回测trade 不管是否成功，异常
                return trade_

        # 开仓处理 遍历各工单(未开仓/已开仓未平仓）
        if open_direction != Direction.NONE:  # 有信号
            # 全部订单 开仓方向进行处理
            for each_trade in working_trades:
                if each_trade.waiting_open:
                    after_open_trades.append(_process_open(each_trade))
                else:
                    # 已开仓的 更新最大浮盈
                    each_trade.record_extreme(high=min1.High, low=min1.Low)
                    after_open_trades.append(each_trade)
        else:  # 没有开仓信号
            for each_trade in working_trades:
                if each_trade.waiting_close:
                    # 已开仓的 更新最大浮盈
                    each_trade.record_extreme(high=min1.High, low=min1.Low)
                after_open_trades.append(each_trade)

        # 获取平仓信号  ==============================================

        def single_get_close_signal(trade_: SimuTrade, ) -> OpenCloseParams:
            """单工单 获取平仓信号"""

            # 止盈平仓 ========================

            if take_profit_type and take_profit_type == 'boll_Min1':  # 一分钟布林带止盈
                # 多头止盈
                if trade_.direction == Direction.LONG:
                    used_price_ = (min1.Last if bl_out_price_type == 'Close' else min1.High)
                    if used_price_ > min1_bolling['out_long']:  # 触发价，==1分钟out_long布林带上
                        return OpenCloseParams(enable=True, type=take_profit_type,  # 止盈标记
                                               price=(min1.Last if bl_out_price_type == 'Close'
                                                      else min1_bolling['out_long']), )
                # 空头止盈
                elif trade_.direction == Direction.SHORT:
                    used_price_ = (min1.Last if bl_out_price_type == 'Close' else min1.Low)
                    if used_price_ < min1_bolling['out_short']:  # 触发价，==1分钟out_short布林带下
                        return OpenCloseParams(enable=True, type=take_profit_type,  # 止盈标记
                                               price=(min1.Last if bl_out_price_type == 'Close'
                                                      else min1_bolling['out_short']))

            # 止损平仓 ==========================================

            if stop_loss_type and stop_loss_type == 'fixed':  # 固定止损
                if trade_.direction == Direction.LONG:
                    # 固定止损的触发价，==开盘价*（1-回落百分比）
                    sl_triggle_price = trade_.open_price * (1 - stop_loss_fix_p / 100)
                    # 判断用的价格
                    used_price_ = (min1.Last if fixed_out_price_type == 'Close' else min1.Low)
                    if used_price_ <= sl_triggle_price:
                        return OpenCloseParams(True, type=stop_loss_type,
                                               price=(used_price_ if fixed_out_price_type == 'Close'
                                                      else sl_triggle_price), )
                elif trade_.direction == Direction.SHORT:  # 平空
                    sl_triggle_price = trade_.open_price * (1 + stop_loss_fix_p / 100)
                    used_price_ = (min1.Last if fixed_out_price_type == 'Close' else min1.High)
                    if used_price_ >= sl_triggle_price:
                        return OpenCloseParams(True, type=stop_loss_type,
                                               price=(used_price_ if fixed_out_price_type == 'Close'
                                                      else sl_triggle_price), )

            # 其他平仓：
            # 日内平仓 反转平仓==用开仓信号反向进行平仓
            if 'opposite_direction' in close_types:
                if open_direction != Direction.NONE and open_direction != trade_.direction:
                    return OpenCloseParams(True, type='opposite_direction', price=min1.Last)


            # 都不符合
            return OpenCloseParams()  # 不平仓

        def batch_get_close_signal() -> OpenCloseParams:
            """批量 获取平仓信号"""

        # 平仓处理
        def _process_close(trade_: SimuTrade,
                           close_signal: OpenCloseParams,  # 平仓信号
                           ) -> SimuTrade:
            try:
                # 平仓
                self.close_trade(trade_,
                                 close_bar=curr_index,
                                 close_type=close_signal.type,
                                 close_price=close_signal.price,
                                 datetime_=min1['datetime'])
            except Exception as e:
                self.log(f'work err: {e} {trade_}', logging.ERROR, exc_info=True)
            finally:
                # 返回全部回测trade 不管是否成功，异常
                return trade_

        # 全部订单 逐一判断平仓
        for each_trade in after_open_trades:
            close_signal_ = single_get_close_signal(each_trade)
            if close_signal_.enable:
                result.append(_process_close(each_trade, close_signal_))
            else:
                result.append(each_trade)

        return result  # body()