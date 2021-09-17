#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略 多均线 趋势
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
import math
import copy

from config import BACK_TESTING_RESULT_DIR
from datahub.fmt import SimuTrade
from constant import Direction, OpenMax
from datahub.indicator import IndicatorCalc as Ind
from datahub.status import BlStatus, MultiBlStatus
from dc.config import DataSourceConfig
from utility import catch_except, timeit
from strategies.template import StrategyTemplate, TradeParams, OpenCloseParams, OpenCloseTypes
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


fibonacci_list = [8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

bolling_line_list = ([5, 10, 15, 20, 25, 30, 45, 60, 90, 120] +
                     fibonacci_list)  # 布林带length选择范围

min1_indicators = {
    'heikin_ashi': lambda df: Ind.heikin_ashi(df),
    'day_extremum': lambda df: Ind.day_extremum(df),
}  # 需要预处理的数据指标
# 加 批量指标
for j in bolling_line_list:
    min1_indicators[f'bolling_sma_{j}'] = eval(f'lambda df: Ind.boll(df, n={j}, ma_type="sma")')
    min1_indicators[f'bolling_ema_{j}'] = eval(f'lambda df: Ind.boll(df, n={j}, ma_type="ema")')

# 进场的布林带周期及其lens -----------------------

# 超参数
hyper_parameter = [{"support_lens": support_lens,
                    "main_len": main_len,
                    'min1_bl_dev': min1_bl_dev,
                    'min1_ma_type': min1_ma_type,
                    'c_price_type': c_price_type,  # 判断与布林带相对位置是用的close价格类型
                    # 进场
                    'open_types': open_types,  # 开仓条件
                    # 'reopen_interval_bars_for_DEV_RANGE': reopen_interval_bars_for_DEV_RANGE,
                    # # 小周期支撑线上开仓需要间隔多少个bar
                    # 'reopen_interval_bars_for_SUPPORT': reopen_interval_bars_for_SUPPORT,
                    # CROSS_SELF_BL再次开仓需要距离上次开仓多少个bar
                    # 'reopen_need_bars_for_cross_self': reopen_need_bars_for_cross_self,
                    # 出场
                    'close_types': close_types,  # 触发平仓类型
                    'stop_loss_start_t': stop_loss_start_t,  # 触发止损 开始时间（小时）
                    'stop_loss_max_f_p': stop_loss_max_f_p,  # 触发止损 最大浮亏门限 %
                    # 'reverse_need_bars': reverse_need_bars,  # 反转破自身 需要空闲多少bar
                    # 'n_bars_is_support_mid': n_bars_is_support_mid,  #
                    # "stop_loss_type": stop_loss_type,
                    # 'out_price_type_fixed': out_price_type_fixed,  # 固定止损出场判断时用什么的价格
                    # 'stop_loss_fixed_p': stop_loss_fixed_p,  # 固定止损比例
                    }
                   for support_lens in [  # 支撑线选用的length
                       [13, 21, 34, 55, 89, 144], ]
                   for main_len in [377, 610, 987][1:2]  # 主布林带length
                   for min1_bl_dev in numpy.arange(1.85, 2.31, 0.05)[3:4]  # 布林带偏移
                   for min1_ma_type in ['ema', 'sma'][1:]
                   for c_price_type in ['Last', 'ha_close_hlcc',  # 价格类型
                                        'ha_close_hlc', 'ha_close'][:1]
                   # 进场
                   for open_types in [  # 进场的价格类型们
                       [  # OpenCloseTypes.DEV_RANGE,
                           # OpenCloseTypes.JUST_MA_TREND,
                           # OpenCloseTypes.MA_TREND_OUTSIDE_BL,
                           OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
                           # OpenCloseTypes.CROSS_SELF_BL_PLUS,
                           # OpenCloseTypes.CROSS_MAIN_BL,
                           # OpenCloseTypes.CROSS_SUPPORT_MID,
                           # OpenCloseTypes.REVERSE_BREAK_SELF,  # 延续单
                       ],
                       # [OpenCloseTypes.CROSS_SELF_BL, ],
                   ]
                   # for reopen_interval_bars_for_DEV_RANGE in [13, 21, 34][:]  # 多均线时有用
                   # for reopen_interval_bars_for_SUPPORT in [0, 5, 13, 21]  # 多均线时有用
                   # for reopen_need_bars_for_cross_self in [0, 5, 13, 34][2:3]
                   # 出场
                   # for reverse_need_bars in [89, 144, 233, 377]
                   # for close_need_bars_for_small in [0, 3, 13, ]
                   # for n_bars_is_support_mid in [13, 21, 34, 55, ]  # 定义支撑线
                   for stop_loss_start_t in [6, 7, 8, 9, 11, 15][2:3]
                   for stop_loss_max_f_p in [0.5, 0.6, 0.75, 1][2:3]  # 触发止损 最大浮亏门限 %
                   for close_types in [
                       [  # OpenCloseTypes.DEV_RANGE,
                           # OpenCloseTypes.JUST_MA_TREND,
                           # OpenCloseTypes.MA_TREND_OUTSIDE_BL,
                           # OpenCloseTypes.DEV_RANGE_TREND,
                           # OpenCloseTypes.CROSS_SUPPORT_MID,
                           # OpenCloseTypes.CROSS_SELF_BL_PLUS,  #
                           # OpenCloseTypes.CROSS_ALL_SUPPORT_MID,
                           # OpenCloseTypes.STOP_LOSS_OVER_TIME,
                           OpenCloseTypes.CROSS_DO_BL,  # 穿越主布林带的对面Line
                           # OpenCloseTypes.REVERSE_BREAK_SELF,  # 反向破自身
                       ],
                       [OpenCloseTypes.CROSS_DO_BL, OpenCloseTypes.STOP_LOSS_OVER_TIME, ],
                   ]
                   # for stop_loss_type in ['fixed', 'boll_Min1'][:1]
                   # for out_price_type_fixed in ['extreme', 'Close'][1:]  # 固定止损出场判断时用什么的价格
                   # for stop_loss_fixed_p in numpy.arange(0.35, 0.351, 0.025)  # 固定止损比例)
                   ]


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
             working_status: Dict,  # 上一轮记录的各种状态
             used_data: Dict[str, pd.DataFrame],  # 已处理好的数据（不同周期的数据 都取最近n个切片）
             params: dict,  # 逻辑判断需要的各个参数 门限值
             ) -> Tuple[List[SimuTrade], Dict]:  # 新的交易状态
        """策略运行 主程序"""

        # 切片数据  curr当前bar  # 最后一个当前bar
        curr_min1 = used_data['min1'].iloc[0]

        # 回测超参数 阈值等 -----------------------------
        support_lens: List[int] = params.get('support_lens')  # 支撑线选用的length
        main_len: int = params.get('main_len')  # 主布林带length
        min1_bl_dev: float = params.get('min1_bl_dev')
        min1_ma_type: str = params.get('min1_ma_type')
        c_price_type: str = params.get('c_price_type')  # 判断是用的实时价的类型
        # 进场
        open_types: List[OpenCloseTypes] = params.get('open_types', )  # 开仓条件
        open_need_bars: int = params.get('open_need_bars', )  #
        reopen_interval_bars_for_dev_range: int = params.get('reopen_interval_bars_for_DEV_RANGE', )  #
        reopen_interval_bars_for_support: int = params.get('reopen_interval_bars_for_SUPPORT', )  #
        reopen_need_bars_for_cross_self: int = params.get('reopen_need_bars_for_cross_self', )  #
        # 出场
        reverse_need_bars: int = params.get('reverse_need_bars', )  # 反向破自身需要多少个bar
        stop_loss_start_t: float = params.get('stop_loss_start_t', )  #
        stop_loss_max_f_p: float = params.get('stop_loss_max_f_p', )  # 止损用最大浮亏门限%
        close_types: List[OpenCloseTypes] = params.get('close_types', )  # 平仓类型
        close_need_bars_for_small: int = params.get('close_need_bars_for_small', )  #
        n_bars_is_support_mid: int = params.get('n_bars_is_support_mid', )  # 定义support
        # stop_loss_type = params.get('stop_loss_type', None)  # 止损判断时用什么的价格
        # out_price_type_fixed = params.get('out_price_type_fixed', )  # 固定止损出场判断时用什么的价格
        # stop_loss_fixed_p = params.get('stop_loss_fixed_p', None)  # 固定止损比例

        # 策略里的交易参数
        trade_p: TradeParams = self.trade_params  # 交易参数
        # 当前处理的bar的index
        curr_index = used_data['min1'].index[-1]

        # 检查数据 若未完全生成，退出这轮
        for len_ in fibonacci_list:
            if math.isnan(curr_min1[f'boll_mid_{min1_ma_type}_{len_}']):
                return working_trades, working_status

        # 上一轮各布林带的状态
        if 'prior_multi_bs' not in working_status:  # 第一次
            _multi_bs: MultiBlStatus = MultiBlStatus(
                min1_ma_type=min1_ma_type,
                dev=min1_bl_dev,
                c_price_type=c_price_type,
            )
            _multi_bs.update_all_bs(curr_index, curr_min1=curr_min1)
            working_status['prior_multi_bs'] = _multi_bs
            return working_trades, working_status
        else:
            prior_multi_bs: MultiBlStatus = copy.deepcopy(working_status['prior_multi_bs'])

        # 生成这轮各布林带的状态
        curr_multi_bs: MultiBlStatus = copy.deepcopy(prior_multi_bs)
        curr_multi_bs.update_all_bs(curr_index, curr_min1=curr_min1)

        # 具体逻辑 =================================================================
        c_price: float = curr_min1[c_price_type]  # 当前使用的价格
        prior_main_direction = working_status.get('main_direction', Direction.NONE)  # 整体波动方向
        prior_range_max = working_status.get('range_max', c_price)  # 波段最大收盘价
        prior_range_min = working_status.get('range_min', c_price)  # 波段最小收盘价
        prior_opened_index = working_status.get('prior_opened_index', -100)  # 上次开仓的bar
        need_stop_loss: bool = working_status.get('need_stop_loss', False)  # 需要止损

        # 预判方向
        threshold = curr_multi_bs.bs[main_len].bl_offset
        predict_direction: Direction = prior_main_direction
        range_max: float = max(c_price, prior_range_max)
        range_min: float = min(c_price, prior_range_min)

        if c_price - prior_range_min > threshold:  # 到达高处
            range_min = c_price
            range_max = c_price
            predict_direction = Direction.SHORT
            if prior_main_direction == Direction.SHORT:
                need_stop_loss = True  # 跨越大方向, 且回到原来方向，就给一次平仓止损机会
        elif prior_range_max - c_price > threshold:  # 到达低处
            range_max = c_price
            range_min = c_price
            predict_direction = Direction.LONG
            if prior_main_direction == Direction.LONG:
                need_stop_loss = True  # 跨越大方向, 且回到原来方向，就给一次平仓止损机会

        # 所有支撑线
        if OpenCloseTypes.JUST_MA_TREND in open_types:
            prior_support_direction: Direction = working_status.get('support_direction', Direction.NONE)
            prior_main_support_len: int = working_status.get('main_support_len', None)
            support_direction, main_support_len = curr_multi_bs.main_support_direction

        # 开仓 -------------------------------------------------
        def batch_get_open_direction() -> OpenCloseParams:
            """
            批量 获取开仓方向
            """

            # 在布林带外 多策略
            if OpenCloseTypes.IN_BL_OUTSIDE_PLUS in open_types:
                main_bs: BlStatus = curr_multi_bs.bs[main_len]
                # 回破小周期 开仓
                # if (prior_open_interval >= reopen_interval_bars_for_support
                #         and cbs.is_out_bl_line):
                if main_bs.is_out_bl_line:
                    for s_len in support_lens:
                        s_direction: Direction = curr_multi_bs.bs[s_len].cross_support_line
                        if Direction.is_opposite(s_direction, main_bs.price_bl):
                            return OpenCloseParams(
                                direction=s_direction,
                                type=OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
                                para=(s_len, main_bs),
                                price=curr_min1.Last, )

            # 动态布林带范围外破趋势
            if (OpenCloseTypes.DEV_RANGE in open_types
                    # 连续开 要隔几个bar
                    and (curr_index - prior_opened_index >= reopen_interval_bars_for_dev_range)
                    # 突破最后一条有效强支撑线
                    and predict_direction != Direction.NONE):
                last_support_line = curr_multi_bs.cross_last_support_line(target_direction=predict_direction)
                if last_support_line:
                    return OpenCloseParams(
                        direction=predict_direction,
                        type=OpenCloseTypes.DEV_RANGE,
                        para=curr_multi_bs.bs[last_support_line],
                        price=c_price, )

            # 仅ma趋势
            if OpenCloseTypes.JUST_MA_TREND in open_types:
                if (prior_support_direction != Direction.NONE
                        and support_direction == Direction.NONE):
                    return OpenCloseParams(
                        direction=Direction.opposite(prior_support_direction),
                        type=OpenCloseTypes.JUST_MA_TREND,
                        para=curr_multi_bs.bs[prior_main_support_len],
                        price=curr_min1.Last, )

            # 布林带+ma趋势
            # if OpenCloseTypes.MA_TREND_OUTSIDE_BL in open_types:
            #     if (curr_multi_bs.bs[main_len])

            # 回破布林带上下轨 开仓
            if OpenCloseTypes.CROSS_DO_BL in open_types:
                main_bs: BlStatus = curr_multi_bs.bs[main_len]
                # if prior_open_interval >= reopen_interval_bars_for_self_bl:
                c_direction: Direction = main_bs.is_break_in_bl
                if (c_direction != Direction.NONE
                        # and cbs.prior_turn_bl_past
                        # and cbs.prior_turn_bl_past >= open_need_bars  # 且过了N个bar
                        # and (curr_index - working_status['prior_opened_index']  # 过n个bar后才能重新开
                        #      >= reopen_need_bars_for_cross_self)
                ):
                    return OpenCloseParams(
                        direction=c_direction,
                        type=OpenCloseTypes.CROSS_DO_BL,
                        para=main_bs,
                        price=curr_min1.Last, )

            # 不符合
            return OpenCloseParams()  # 不开仓

        # def reverse_break_self_open_signal(trade_: SimuTrade, ) -> OpenCloseParams:
        #     """
        #     反向破自身的延续单 开仓
        #     """
        #     direction, len_ = trade_.open_para
        #     cbs: BlStatus = curr_prior_bs[len_]
        #
        #     if direction == cbs.is_break_in_bl:
        #         return OpenCloseParams(direction=direction,
        #                                type=OpenCloseTypes.REVERSE_BREAK_MAIN_BL,
        #                                para=prior_multi_bs[len_],
        #                                price=c_price, )
        #     # 不符合
        #     return OpenCloseParams()  # 不开仓

        # 开仓逻辑开始 ======================================
        after_open_processed_trades, result = [], []

        def _process_open(trade_: SimuTrade, open_signal: OpenCloseParams):
            """统一开仓处理"""
            try:
                # 没开仓 且 满足开仓 用SimTrade.set_open
                trade_.set_open(direction_=open_signal.direction,
                                bar=curr_index,  # 当前bar
                                price=open_signal.price,  # 当前价
                                amount=trade_p.once_lots,  # 开仓量
                                datetime_=curr_min1['datetime'],  # 开仓时间
                                open_type=open_signal.type,
                                open_para=open_signal.para,
                                )
                # todo 全局变量 n个bar内不再买

                working_status['prior_opened_index'] = curr_index  # 更新开仓bar

            except Exception as e:
                self.log(f'work err: {e}', logging.ERROR, exc_info=True)
            finally:
                # 返回全部回测trade 不管是否成功，异常
                return trade_

        # 获取公共开仓信号 可以统一获取
        open_batch_signal: OpenCloseParams = batch_get_open_direction()

        # 开仓处理 遍历各工单(未开仓/已开仓未平仓）
        for a_trade in working_trades:
            if a_trade.waiting_open:
                # 普通空白单
                if open_batch_signal.is_valid:  # 有信号 开仓
                    a_trade = _process_open(a_trade, open_batch_signal)
            else:
                # 已开仓的 只更新最大浮盈
                a_trade.record_extreme(high=curr_min1.High, low=curr_min1.Low,
                                       bar_index=curr_index)
            # 每一个工单都保存起来
            after_open_processed_trades.append(a_trade)

        # 平仓  ==============================================================

        def get_close_signal_DEV_RANGE_TREND() -> OpenCloseParams:
            """止损 布林带大区域逆转+趋势（持仓的反方向有支撑线）"""
            if need_stop_loss:
                support_line = curr_multi_bs.has_support_line(predict_direction)
                if support_line:
                    return OpenCloseParams(
                        direction=predict_direction,  # 平 原来开仓同方向的单
                        type=OpenCloseTypes.DEV_RANGE_TREND,
                        para=curr_multi_bs.bs[support_line],
                        price=c_price, )

        def batch_get_close_signal() -> OpenCloseParams:
            """
            批量 获取平仓信号
            """
            # 止盈 布林带大区域，PS:取同类型开仓信息反向就好
            if OpenCloseTypes.DEV_RANGE in close_types:
                if open_batch_signal.type == OpenCloseTypes.DEV_RANGE:
                    params_ = open_batch_signal
                    params_.direction = Direction.opposite(open_batch_signal.direction)
                    return params_

            # 仅ma趋势 平仓
            if OpenCloseTypes.JUST_MA_TREND in close_types:
                if open_batch_signal.type == OpenCloseTypes.JUST_MA_TREND:
                    params_ = open_batch_signal
                    params_.direction = Direction.opposite(open_batch_signal.direction)
                    return params_
                elif (prior_support_direction == Direction.NONE
                      and support_direction != Direction.NONE):
                    return OpenCloseParams(
                        direction=(Direction.LONG
                                   if support_direction == Direction.SHORT
                                   else Direction.SHORT),
                        type=OpenCloseTypes.JUST_MA_TREND,
                        para=curr_multi_bs.bs[main_support_len],
                        price=curr_min1.Last, )

            # 回破布林带上下轨 平仓
            if OpenCloseTypes.CROSS_DO_BL in close_types:
                # 突破（开仓）的方向
                cbs: BlStatus = curr_multi_bs.bs[main_len]
                if cbs.is_break_in_bl != Direction.NONE:
                    return OpenCloseParams(
                        direction=(Direction.LONG
                                   if cbs.is_break_in_bl == Direction.SHORT
                                   else Direction.SHORT),
                        type=OpenCloseTypes.CROSS_DO_BL,
                        para=cbs,
                        price=curr_min1.Last, )

            # 不满足
            return OpenCloseParams()

        def single_get_close_signal(trade_: SimuTrade, ) -> OpenCloseParams:
            """
            单工单 获取平仓信号
            """
            # 超时止损
            if OpenCloseTypes.STOP_LOSS_OVER_TIME in close_types:
                # 超时 + 实时最大浮亏大于门限 + 布林带外
                if curr_index - trade_.open_bar > stop_loss_start_t * 60:
                    max_ = max(trade_.highest_v, curr_min1.Last)
                    min_ = min(trade_.lowest_v, curr_min1.Last)
                    # 最大浮亏 %
                    fl = ((max_ - trade_.open_price) / trade_.open_price
                          if trade_.direction == Direction.SHORT
                          else (trade_.open_price - min_) / trade_.open_price)
                    # # 实时盈利
                    # real_profit = (curr_min1.Last - trade_.open_price
                    #                if trade_.direction == Direction.LONG
                    #                else trade_.open_price - curr_min1.Last)
                    if ((fl * 100 > stop_loss_max_f_p)
                            and Direction.is_opposite(
                                curr_multi_bs.bs[main_len].price_bl, trade_.direction)):
                        return OpenCloseParams(
                            direction=trade_.direction,
                            type=OpenCloseTypes.STOP_LOSS_OVER_TIME,
                            para=(curr_index - trade_.open_bar, fl * 100),
                            price=curr_min1.Last, )
            # 不满足
            return OpenCloseParams()

        # 平仓处理
        def _process_close(trade_: SimuTrade,
                           close_signal: OpenCloseParams,  # 平仓信号
                           ) -> SimuTrade:
            try:
                # 平仓
                if trade_.waiting_close:
                    self.close_trade(trade_,
                                     close_bar=curr_index,
                                     close_type=close_signal.type,
                                     close_price=close_signal.price,
                                     datetime_=curr_min1['datetime'],
                                     close_para=close_signal.para, )
                    nonlocal need_stop_loss
                    need_stop_loss = False  # 清空，一次反转只平仓一次

            except Exception as e:
                self.log(f'work err: {e} {trade_}', logging.ERROR, exc_info=True)
            finally:
                # 返回全部回测trade 不管是否成功，异常
                return trade_

        # 全部订单 逐一判断平仓
        drt_close_signal: OpenCloseParams = (get_close_signal_DEV_RANGE_TREND()
                                             if OpenCloseTypes.DEV_RANGE_TREND in close_types
                                             else None)

        batch_close_signal: OpenCloseParams = batch_get_close_signal()
        for a_trade in after_open_processed_trades:
            if a_trade.waiting_close:
                # 统一产生的平仓止损信号
                if drt_close_signal and drt_close_signal.is_valid:
                    a_trade = _process_close(a_trade, drt_close_signal)

            if a_trade.waiting_close:
                # 其他公用的平仓信号
                if batch_close_signal.direction == a_trade.direction:  # 是这个工单需要的平仓方向
                    a_trade = _process_close(a_trade, batch_close_signal)

            if a_trade.waiting_close:
                single_close_signal: OpenCloseParams = single_get_close_signal(a_trade)
                if single_close_signal.is_valid:
                    a_trade = _process_close(a_trade, single_close_signal)

            # 收集回来
            result.append(a_trade)

        # 更新全局状态 =================================================
        working_status['prior_multi_bs'] = curr_multi_bs
        working_status['main_direction'] = predict_direction
        working_status['range_max'] = range_max
        working_status['range_min'] = range_min
        working_status['need_stop_loss'] = need_stop_loss
        if OpenCloseTypes.JUST_MA_TREND in open_types:
            working_status['support_direction'] = support_direction
            working_status['main_support_len'] = main_support_len

        return result, working_status  # body()
