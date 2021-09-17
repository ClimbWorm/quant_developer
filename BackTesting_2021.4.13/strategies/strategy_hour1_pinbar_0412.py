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
from datahub.bar import Bar, PinBar
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
hyper_parameter = [
    {
        # 'min1_ma_type': min1_ma_type,
        'c_price_type': c_price_type,  # 判断与布林带相对位置是用的close价格类型
        'zzp_reversal': zzp_reversal,  # zzp反转%
        # "do_bl_len": do_bl_len,
        # 'do_bl_dev': do_bl_dev,
        # "base_bl_len": base_bl_len,
        # 'base_bl_dev': base_bl_dev,
        # "support_lens": support_lens,
        # 进场
        'open_types': open_types,  # 开仓条件
        # 出场
        'close_types': close_types,  # 触发平仓类型
        # 'fixed_stop_loss_p': fixed_stop_loss_p,  # 固定止损%
    }
    # for min1_ma_type in ['ema', 'sma'][1:]
    for c_price_type in ['Last', 'ha_close_hlcc',  # 价格类型
                         'ha_close_hlc', 'ha_close'][:1]
    for zzp_reversal in numpy.arange(0.8, 1.21, 0.2)  #
    # for do_bl_len in [21, 34, 55, 89, 144, 233, 377, 610, ][4:5]  # 操作布林带length
    # for do_bl_dev in numpy.arange(1.7, 2.31, 0.1)[:]  # 操作布林带偏移
    # for base_bl_len in [377, 610, 987][1:2]  # 参考布林带length
    # for base_bl_dev in numpy.arange(1.6, 2.51, 0.1)[4:5]  # 参考布林带偏移
    # for support_lens in [  # 支撑线选用的length
    #     # [21, 34, 55, 89, 144],
    #     [13, 21, 34, 55, 89, 144][:2],
    # ]
    # 进场
    for open_types in [  # 进场的价格类型们
        [  # OpenCloseTypes.DEV_RANGE,
            OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
            # OpenCloseTypes.REVERSE_BREAK_SELF,  # 延续单
        ],
        # [OpenCloseTypes.TOUCH_BL_OUTSIDE_PLUS, ],
        # [OpenCloseTypes.CROSS_MAIN_BL, ],
    ]
    # 出场
    for close_types in [
        [  # OpenCloseTypes.DEV_RANGE,
            OpenCloseTypes.CROSS_DO_BL,  # 穿越操作布林带的对面Line
            # OpenCloseTypes.CROSS_MAIN_BL_BASE_MAIN,  #
            # OpenCloseTypes.REVERSE_BREAK_SELF,  # 反向破自身
        ],
    ]
    # for fixed_stop_loss_p in numpy.arange(0.1, 0.31, 0.1)  # 固定止损%
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
                'min60': set_bt_data_setting(
                    data_source=data_source,
                    new_bar_interval=Sec.HOUR1.value,  # 新bar 周期
                    indicators=min1_indicators,
                ),
            },  # self.data_sets

            # 回测需要的超参数 list
            hyper_parameter=hyper_parameter[:1],

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
        curr_min60 = used_data['min60'].iloc[0]

        # 回测超参数 阈值等 -----------------------------
        c_price_type: str = params.get('c_price_type')  # 判断是用的实时价的类型
        zzp_reversal: float = params.get('zzp_reversal')  #
        # do_bl_len: int = params.get('do_bl_len')  # 操作布林带length
        # do_bl_dev: int = params.get('do_bl_dev')  #
        # base_bl_len: int = params.get('base_bl_len')  #
        # base_bl_dev: int = params.get('base_bl_dev')  #
        # support_lens: List[int] = params.get('support_lens')  # 支撑线选用的length
        # 进场
        open_types: List[OpenCloseTypes] = params.get('open_types', )  # 开仓条件
        # 出场
        close_types: List[OpenCloseTypes] = params.get('close_types', )  # 平仓类型

        # support_n = {
        #     # 13: 13, 21: 21, 34: 34, 55: 55, 89: 89, 144: 144, 233: 233, 377: 377, 610: 610
        # }  # 默认支撑线n值
        # in_bl_t_n = {}  # 在布林带内持续n个bar算超限

        # 策略里的交易参数
        trade_p: TradeParams = self.trade_params  # 交易参数

        # 当前处理的bar的index
        curr_index = used_data['min60'].index[-1]
        curr_bar: Bar = Bar(index=used_data['min60'].index[-1],
                            open=curr_min60.Open,
                            high=curr_min60.High,
                            low=curr_min60.Low,
                            close=curr_min60.Last,
                            volume=curr_min60.Volume,
                            period=Sec.HOUR1,)

        # 检查数据 若未完全生成，退出这轮
        # for len_ in fibonacci_list:
        #     if math.isnan(curr_min1[f'boll_mid_{min1_ma_type}_{len_}']):
        #         return working_trades, working_status

        pin_bar = PinBar(amp_p_threshold=0.5)
        is_pin_bar = pin_bar.is_pin_bar(curr_bar=curr_bar)
        if is_pin_bar:
            print(is_pin_bar)

        # 上一轮各布林带的状态
        # if 'multi_bs' not in working_status:  # 第一次
        #     _multi_bs: MultiBlStatus = MultiBlStatus(
        #         min1_ma_type=min1_ma_type,
        #         c_price_type=c_price_type,
        #         all_lines=fibonacci_list,
        #     )
        #     # 先初始化各布林带的参数
        #     for len_ in _multi_bs.all_lines:
        #         _multi_bs.bs[len_] = BlStatus(  # 第一次
        #             len_=len_,
        #             dev=(base_bl_dev if len_ == base_bl_len
        #                  else (do_bl_dev if len_ == do_bl_len else 2.0)),  # 偏移
        #             n_for_support=support_n.get(len_, len_),
        #             n_for_in_bl_t=in_bl_t_n.get(len_, int(1.5 * len_ ** 0.5)),
        #         )
        #
        #     _multi_bs.update_all_bs(curr_index, curr_min1=curr_min1)
        #     working_status['multi_bs'] = _multi_bs
        #     return working_trades, working_status
        # else:
        #     prior_multi_bs: MultiBlStatus = copy.deepcopy(working_status['multi_bs'])

        # # 上一轮价格运动状态
        # if 'price_status' not in working_status:  # 第一次
        #     working_status['price_status'] = PriceStatus(
        #         open_price=curr_min1.Last,
        #     )
        #     return working_trades, working_status
        # else:
        #     prior_price_status: PriceStatus = copy.deepcopy(working_status['price_status'])

        # 生成这轮各布林带的状态
        # curr_multi_bs: MultiBlStatus = copy.deepcopy(prior_multi_bs)
        # curr_multi_bs.update_all_bs(curr_index, curr_min1=curr_min1)

        # curr_price_status: PriceStatus = copy.deepcopy(prior_price_status)
        # curr_price_status.update_dynamic(curr_min1.Last)

        # 具体逻辑 =================================================================
        c_price: float = curr_min60[c_price_type]  # 当前使用的价格
        # do_bs: BlStatus = curr_multi_bs.bs[do_bl_len]
        # base_bs: BlStatus = curr_multi_bs.bs[base_bl_len]
        # 基础方向 参考布林带最近是破上轨还是下轨
        # base_direction: Direction = base_bs.price_bl if base_bs.is_out_bl_line else base_bs.prior_turn_p_bl

        # 开仓 -------------------------------------------------
        def batch_get_open_direction() -> OpenCloseParams:
            """
            批量 获取开仓方向
            """

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
                    datetime_=curr_min60['datetime'],  # 开仓时间
                    open_type=open_signal.type,
                    open_para=open_signal.para,
                    price_status=PriceStatus(
                        sl_range_threshold_p=params.get('sl_range_threshold_p', ),
                        sl_range_draw_down_p=params.get('sl_range_draw_down_p', ),
                        base_direction=direction,
                        open_price=curr_min60.Last,
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
                a_trade.record_extreme(high=curr_min60.High, low=curr_min60.Low,
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

            # 触摸到布林带外 多策略
            # if OpenCloseTypes.TOUCH_BL_OUTSIDE_PLUS in close_types:
            #     if (not curr_multi_bs.has_support_line(base_p_bl)
            #             and Direction.is_opposite(base_p_bl, curr_multi_bs.bs[8].price_mid)):
            #         return OpenCloseParams(
            #             direction=base_p_bl,
            #             type=OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
            #             para=main_bs,
            #             price=curr_min1.Last, )

            # 不满足
            return OpenCloseParams()

        def single_get_close_signal(trade_: SimuTrade, ) -> OpenCloseParams:
            """
            单工单 获取平仓信号
            """
            # # 固定止损
            # if OpenCloseTypes.FIXED_STOP_LOSS in close_types:
            #     p = params.get('fixed_stop_loss_p')  # 固定止损%
            #     if ((trade_.direction == Direction.LONG
            #          and c_price <= trade_.open_price * (1 - p / 100))
            #             or (trade_.direction == Direction.SHORT
            #                 and c_price >= trade_.open_price * (1 + p / 100))):
            #         return OpenCloseParams(
            #             direction=trade_.direction,
            #             type=OpenCloseTypes.FIXED_STOP_LOSS,
            #             para=p,
            #             price=c_price, )
            #

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
                                     datetime_=curr_min60['datetime'],
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

            if a_trade.waiting_close:
                single_close_signal: OpenCloseParams = single_get_close_signal(a_trade)
                if single_close_signal.is_valid:
                    a_trade = _process_close(a_trade, single_close_signal)

            # 收集回来
            result.append(a_trade)

        # 更新全局状态 =================================================
        # working_status['multi_bs'] = curr_multi_bs
        # working_status['price_status'] = curr_price_status

        return result, working_status  # body()