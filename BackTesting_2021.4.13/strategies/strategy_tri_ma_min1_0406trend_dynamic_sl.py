#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略 多均线 趋势
# 进程 出场都用1分钟
# trend + 动态dev止损

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
from datahub.status import BlStatus, MultiBlStatus, BlDevStatus, PriceStatus
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
                    # 'reopen_need_bars_for_cross_self': reopen_need_bars_for_cross_self,
                    # 出场
                    # 'sl_start': sl_start,  #
                    # 'sl_range_threshold_p': sl_range_threshold_p,  #
                    # 'sl_range_draw_down_p': sl_range_draw_down_p,  #
                    'close_types': close_types,  # 触发平仓类型
                    }
                   for support_lens in [  # 支撑线选用的length
                       # [13, 21, 34, 55, 89],
                       # [21, 34, 55, 89, 144],
                       [13, 21, 34, 55, 89, 144][:2],
                   ]
                   for main_len in [34, 55, 89, 144, 233, 377, 610, ][:6]  # 主布林带length
                   for min1_bl_dev in numpy.arange(1.85, 2.351, 0.05)[:]  # 布林带偏移
                   for min1_ma_type in ['ema', 'sma'][:]
                   for c_price_type in ['Last', 'ha_close_hlcc',  # 价格类型
                                        'ha_close_hlc', 'ha_close'][:]
                   # 进场
                   for open_types in [  # 进场的价格类型们
                       [  # OpenCloseTypes.DEV_RANGE,
                           # OpenCloseTypes.JUST_MA_TREND,
                           # OpenCloseTypes.MA_TREND_OUTSIDE_BL,
                           # OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
                           # OpenCloseTypes.TOUCH_BL_AND_CROSS_MID,
                           # OpenCloseTypes.TOUCH_BL_OUTSIDE_PLUS,
                           # OpenCloseTypes.CROSS_SELF_BL_PLUS,
                           # OpenCloseTypes.CROSS_MAIN_BL,
                           OpenCloseTypes.CROSS_BL_ACCORDING_TO_BASE,  #
                           # OpenCloseTypes.CROSS_SUPPORT_MID,
                           # OpenCloseTypes.REVERSE_BREAK_SELF,  # 延续单
                       ],
                       # [OpenCloseTypes.TOUCH_BL_OUTSIDE_PLUS, ],
                       # [OpenCloseTypes.CROSS_MAIN_BL, ],
                   ]
                   # for reopen_interval_bars_for_DEV_RANGE in [13, 21, 34][:]  # 多均线时有用
                   # for reopen_interval_bars_for_SUPPORT in [0, 5, 13, 21]  # 多均线时有用
                   # 出场
                   # for sl_start in [55, 89, 144, 233, 377, 610][:]
                   # for sl_range_threshold_p in numpy.arange(0, 0.5, 0.1)[:3]  # 回撤前的反弹 阈值%
                   # for sl_range_draw_down_p in numpy.arange(0.5, 0.91, 0.1)  # 回撤止损值 %
                   for close_types in [
                       [  # OpenCloseTypes.DEV_RANGE,
                           # OpenCloseTypes.DYNAMIC_DEV,  # 动态布林带区域
                           # OpenCloseTypes.JUST_MA_TREND,
                           # OpenCloseTypes.MA_TREND_OUTSIDE_BL,
                           # OpenCloseTypes.DEV_RANGE_TREND,
                           # OpenCloseTypes.CROSS_SUPPORT_MID,
                           # OpenCloseTypes.CROSS_SELF_BL_PLUS,  #
                           # OpenCloseTypes.CROSS_ALL_SUPPORT_MID,
                           # OpenCloseTypes.STOP_LOSS_OVER_TIME,
                           # OpenCloseTypes.TRIGGER,
                           # OpenCloseTypes.TOUCH_BL_AND_CROSS_MID,  # 止损
                           # OpenCloseTypes.TOUCH_BL_OUTSIDE_PLUS  # 止盈
                           OpenCloseTypes.CROSS_DO_BL,  # 穿越主布林带的对面Line
                           # OpenCloseTypes.CROSS_MAIN_BL_BASE_MAIN,  #
                           # OpenCloseTypes.REVERSE_BREAK_SELF,  # 反向破自身
                       ],
                       # [OpenCloseTypes.DYNAMIC_DEV,OpenCloseTypes.STOP_LOSS_OVER_TIME, ]
                   ]
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
        close_types: List[OpenCloseTypes] = params.get('close_types', )  # 平仓类型
        sl_start: int = params.get('sl_start', )  # 几个bar后开始激活平仓策略
        n_bars_is_support_mid: int = params.get('n_bars_is_support_mid', )  # 定义support

        # 策略里的交易参数
        trade_p: TradeParams = self.trade_params  # 交易参数
        # 当前处理的bar的index
        curr_index = used_data['min1'].index[-1]

        # 检查数据 若未完全生成，退出这轮
        for len_ in fibonacci_list:
            if math.isnan(curr_min1[f'boll_mid_{min1_ma_type}_{len_}']):
                return working_trades, working_status

        # 上一轮各布林带的状态
        if 'multi_bs' not in working_status:  # 第一次
            _multi_bs: MultiBlStatus = MultiBlStatus(
                min1_ma_type=min1_ma_type,
                dev=min1_bl_dev,
                c_price_type=c_price_type,
            )
            _multi_bs.update_all_bs(curr_index, curr_min1=curr_min1)
            working_status['multi_bs'] = _multi_bs
            return working_trades, working_status
        else:
            prior_multi_bs: MultiBlStatus = copy.deepcopy(working_status['multi_bs'])

        # # 上一轮价格运动状态
        # if 'price_status' not in working_status:  # 第一次
        #     working_status['price_status'] = PriceStatus(
        #         open_price=curr_min1.Last,
        #     )
        #     return working_trades, working_status
        # else:
        #     prior_price_status: PriceStatus = copy.deepcopy(working_status['price_status'])

        # 生成这轮各布林带的状态
        curr_multi_bs: MultiBlStatus = copy.deepcopy(prior_multi_bs)
        curr_multi_bs.update_all_bs(curr_index, curr_min1=curr_min1)

        # curr_price_status: PriceStatus = copy.deepcopy(prior_price_status)
        # curr_price_status.update_dynamic(curr_min1.Last)

        # 具体逻辑 =================================================================
        c_price: float = curr_min1[c_price_type]  # 当前使用的价格
        main_bs: BlStatus = curr_multi_bs.bs[main_len]
        # 最近是破上轨还是下轨
        base_p_bl: Direction = main_bs.price_bl if main_bs.is_out_bl_line else main_bs.prior_turn_p_bl

        # prior_opened_index = working_status.get('prior_opened_index', -100)  # 上次开仓的bar

        # 开仓 -------------------------------------------------
        def batch_get_open_direction() -> OpenCloseParams:
            """
            批量 获取开仓方向
            """
            main_bs: BlStatus = curr_multi_bs.bs[main_len]
            # # 触摸到布林带外 破中线
            # if OpenCloseTypes.TOUCH_BL_AND_CROSS_MID in open_types:
            #     cross_mid: Direction = main_bs.is_cross_mid
            #
            #     if cross_mid != Direction.NONE and Direction.is_opposite(cross_mid, base_p_bl):
            #         return OpenCloseParams(
            #             direction=cross_mid,
            #             type=OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
            #             para=main_bs,
            #             price=curr_min1.Last, )

            # 基于交易主方向的 强回穿主布林带
            if OpenCloseTypes.CROSS_BL_ACCORDING_TO_BASE in open_types:
                base_bs: BlStatus = curr_multi_bs.bs[610]
                cross_bl: Direction = main_bs.is_break_in_bl
                # 最近交易主方向是破上轨还是下轨
                base_p_bl: Direction = base_bs.price_bl if base_bs.is_out_bl_line else base_bs.prior_turn_p_bl
                if cross_bl != Direction.NONE and cross_bl == base_p_bl:
                    return OpenCloseParams(
                        direction=cross_bl,
                        type=OpenCloseTypes.CROSS_BL_ACCORDING_TO_BASE,
                        para=main_bs,
                        price=curr_min1.Last, )

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
                direction = open_signal.direction
                # # 不一样的话，强行更新dev status
                # if direction != curr_dev_status.predict_direction:
                #     curr_dev_status.change_predict_direction(direction)

                # 没开仓 且 满足开仓 用SimTrade.set_open
                trade_.set_open(
                    direction_=direction,
                    bar=curr_index,  # 当前bar
                    price=open_signal.price,  # 当前价
                    amount=trade_p.once_lots,  # 开仓量
                    datetime_=curr_min1['datetime'],  # 开仓时间
                    open_type=open_signal.type,
                    open_para=open_signal.para,
                    price_status=PriceStatus(
                        sl_range_threshold_p=params.get('sl_range_threshold_p', ),
                        sl_range_draw_down_p=params.get('sl_range_draw_down_p', ),
                        base_direction=direction,
                        open_price=curr_min1.Last,
                        # ref=(curr_price_status.dynamic_min
                        #      if direction == Direction.LONG
                        #      else curr_price_status.dynamic_max)  # 止损参考价格
                    ),
                )
                # todo 全局变量 n个bar内不再买

                # working_status['prior_opened_index'] = curr_index  # 更新开仓bar

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

        # 是否有订单需要平仓 没有直接退出
        has_waiting_close_trades: bool = False
        for a_trade in after_open_processed_trades:
            if a_trade.waiting_close:
                has_waiting_close_trades = True
                break

        def batch_get_close_signal() -> OpenCloseParams:
            """
            批量 获取平仓信号
            """
            # 触摸到布林带外 破中线
            if OpenCloseTypes.TOUCH_BL_AND_CROSS_MID in close_types:
                cross_mid: Direction = main_bs.is_cross_mid
                if cross_mid != Direction.NONE and cross_mid == base_p_bl:
                    return OpenCloseParams(
                        direction=Direction.opposite(cross_mid),
                        type=OpenCloseTypes.TOUCH_BL_AND_CROSS_MID,
                        para=main_bs,
                        price=curr_min1.Last, )

            # 触摸到布林带外 多策略
            # if OpenCloseTypes.TOUCH_BL_OUTSIDE_PLUS in close_types:
            #     if (not curr_multi_bs.has_support_line(base_p_bl)
            #             and Direction.is_opposite(base_p_bl, curr_multi_bs.bs[8].price_mid)):
            #         return OpenCloseParams(
            #             direction=base_p_bl,
            #             type=OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
            #             para=main_bs,
            #             price=curr_min1.Last, )

            # # 基于交易主方向的 强回穿主布林带
            # if OpenCloseTypes.CROSS_MAIN_BL_BASE_MAIN in close_types:
            #     base_bs: BlStatus = curr_multi_bs.bs[610]
            #     cross_bl: Direction = main_bs.is_break_in_bl
            #     # 最近交易主方向是破上轨还是下轨
            #     base_p_bl: Direction = base_bs.price_bl if base_bs.is_out_bl_line else base_bs.prior_turn_p_bl


            # 回破主布林带上下轨 平仓
            if OpenCloseTypes.CROSS_DO_BL in close_types:
                # 突破（开仓）的方向
                if main_bs.is_break_in_bl != Direction.NONE:
                    return OpenCloseParams(
                        direction=(Direction.LONG
                                   if main_bs.is_break_in_bl == Direction.SHORT
                                   else Direction.SHORT),
                        type=OpenCloseTypes.CROSS_DO_BL,
                        para=main_bs,
                        price=curr_min1.Last, )

            # 不满足
            return OpenCloseParams()

        def single_get_close_signal(trade_: SimuTrade, ) -> OpenCloseParams:
            """
            单工单 获取平仓信号
            """
            if OpenCloseTypes.TRIGGER in close_types:
                trade_.price_status.update(curr_min1.Last)  # 先更新
                if (curr_index - trade_.open_bar > sl_start
                        # and trade_.price_status.reached_range_draw_down(curr_min1.Last)
                ):
                    ref = trade_.price_status.ref  # 止损位
                    over: bool = (curr_min1.Last < ref
                                  if trade_.direction == Direction.LONG
                                  else curr_min1.Last > ref)

                    if (over and not curr_multi_bs.has_support_line(trade_.direction)
                            and Direction.is_opposite(trade_.direction,
                                                      curr_multi_bs.bs[8].price_mid)):
                        return OpenCloseParams(
                            direction=trade_.direction,
                            type=OpenCloseTypes.TRIGGER,
                            para=trade_.price_status,
                            price=c_price, )

            # 动态布林带
            # if OpenCloseTypes.DYNAMIC_DEV in close_types:
            # ds: BlDevStatus = trade_.dev_status
            #
            # need_tp1: bool = (ds.tp1_triggered != Direction.NONE
            #                   and not curr_multi_bs.has_support_line(trade_.direction)
            #                   and Direction.is_opposite(trade_.direction,
            #                                             curr_multi_bs.bs[8].price_mid))
            #
            # # 判断止盈 止损
            # if need_tp1 or ds.need_tp2 != Direction.NONE or ds.need_sl != Direction.NONE:
            #     return OpenCloseParams(
            #         direction=trade_.direction,
            #         type=('DYNAMIC_DEV_TP1' if need_tp1
            #               else ('DYNAMIC_DEV_TP2' if ds.need_tp2 != Direction.NONE
            #                     else 'DYNAMIC_DEV_SL')),
            #         para=curr_multi_bs.bs[main_len],
            #         price=c_price, )

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

            except Exception as e:
                self.log(f'work err: {e} {trade_}', logging.ERROR, exc_info=True)
            finally:
                # 返回全部回测trade 不管是否成功，异常
                return trade_

        # 全部订单 逐一判断平仓

        if has_waiting_close_trades:
            batch_close_signal: OpenCloseParams = batch_get_close_signal()
        else:
            batch_close_signal: OpenCloseParams = OpenCloseParams()

        for a_trade in after_open_processed_trades:
            if a_trade.waiting_close:
                # 其他公用的平仓信号
                if batch_close_signal.direction == a_trade.direction:  # 是这个工单需要的平仓方向
                    a_trade = _process_close(a_trade, batch_close_signal)

            # if a_trade.waiting_close:
            #     single_close_signal: OpenCloseParams = single_get_close_signal(a_trade)
            #     if single_close_signal.is_valid:
            #         a_trade = _process_close(a_trade, single_close_signal)

            # 收集回来
            result.append(a_trade)

        # 更新全局状态 =================================================
        working_status['multi_bs'] = curr_multi_bs
        # working_status['price_status'] = curr_price_status

        return result, working_status  # body()
