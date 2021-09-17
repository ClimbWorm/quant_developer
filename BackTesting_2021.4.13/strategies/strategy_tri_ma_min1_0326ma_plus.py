#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 策略 新3均线 基于1分钟 ma 不限制开仓 趋势反向开仓 不带日内平仓
# 加强版
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

from config import BACK_TESTING_RESULT_DIR
from datahub.fmt import SimuTrade
from constant import Direction, OpenMax
from datahub.indicator import IndicatorCalc as Ind
from datahub.status import BlStatus
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

lens_p8 = [{"min1_bl_lens": 8,  # 一分钟线 布林带周期
            'open_need_bars': open_need_bars, }  # 开仓需要多少个bar
           for open_need_bars in range(2)]

lens_p13 = [{"min1_bl_lens": 13,
             'open_need_bars': open_need_bars, }  # 开仓需要多少个bar
            for open_need_bars in range(4)]

lens_p21 = [{"min1_bl_lens": 21,
             'open_need_bars': open_need_bars, }  # 开仓需要多少个bar
            for open_need_bars in range(6)]

lens_p34 = [{"min1_bl_lens": 34,
             'open_need_bars': open_need_bars, }  # 开仓需要多少个bar
            for open_need_bars in range(8)]

lens_p55 = [{"min1_bl_lens": 55,
             'open_need_bars': open_need_bars, }  # 开仓需要多少个bar
            for open_need_bars in range(10)]

lens_p89et = [{"min1_bl_lens": min1_bl_lens,
               'open_need_bars': open_need_bars, }  # 开仓需要多少个bar
              for min1_bl_lens in [89, 144, 233, 377, 610, ][-1:]
              for open_need_bars in [15, ]]

all_lens_bars = lens_p89et  # lens_p8 +  + lens_p13 + lens_p21 + lens_p34 + lens_p55

# 超参数
hyper_parameter_base = [{'min1_bl_dev': min1_bl_dev,
                         'min1_ma_type': min1_ma_type,
                         'pos_use_close_type': pos_use_close_type,  # 判断与布林带相对位置是用的close价格类型
                         # 进场
                         'open_types': open_types,  # 开仓条件
                         # 'reopen_interval_bars_for_SELF_BL': reopen_interval_bars_for_SELF_BL,
                         # # 小周期支撑线上开仓需要间隔多少个bar
                         # 'reopen_interval_bars_for_SUPPORT': reopen_interval_bars_for_SUPPORT,
                         'support_lens_for_open': support_lens_for_open,  #
                         # CROSS_SELF_BL再次开仓需要距离上次开仓多少个bar
                         # 'reopen_need_bars_for_cross_self': reopen_need_bars_for_cross_self,
                         # 出场
                         'close_types': close_types,  # 触发平仓类型
                         # 'reverse_need_bars': reverse_need_bars,  # 反转破自身 需要空闲多少bar
                         # 'n_bars_is_support_mid': n_bars_is_support_mid,  #
                         # "stop_loss_type": stop_loss_type,
                         # 'out_price_type_fixed': out_price_type_fixed,  # 固定止损出场判断时用什么的价格
                         # 'stop_loss_fixed_p': stop_loss_fixed_p,  # 固定止损比例
                         }
                        #
                        # for min1_bl_lens in [(8,), (13,), (21,), (34,), (55,), (89,),
                        #                      (144,), (233,), (377,), (610,),
                        #                      ][::-1]
                        for min1_bl_dev in numpy.arange(1.95, 2.351, 0.05)[:]  # 布林带偏移
                        for min1_ma_type in ['ema', 'sma'][1:]
                        for pos_use_close_type in ['Last', 'ha_close_hlcc',  # 价格类型
                                                   'ha_close_hlc', 'ha_close'][:1]
                        # 进场
                        for open_types in [  # 进场的价格类型们
                            [OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
                             # OpenCloseTypes.CROSS_SELF_BL_PLUS,
                             OpenCloseTypes.CROSS_DO_BL,
                             # OpenCloseTypes.CROSS_SUPPORT_MID,
                             # OpenCloseTypes.REVERSE_BREAK_SELF,  # 延续单
                             ],
                            # [OpenCloseTypes.CROSS_SELF_BL, ],
                        ]
                        # for reopen_interval_bars_for_SELF_BL in [0, 5, 13, 21]  # 多均线时有用
                        # for reopen_interval_bars_for_SUPPORT in [0, 5, 13, 21]  # 多均线时有用
                        for support_lens_for_open in [[13, 21, 34, 55, ],
                                                      # [13, 21, 34,]
                                                      ]  # 支撑线length
                        # for reopen_need_bars_for_cross_self in [0, 5, 13, 34][2:3]
                        # 出场
                        # for reverse_need_bars in [89, 144, 233, 377]
                        # for close_need_bars_for_small in [0, 3, 13, ]
                        # for n_bars_is_support_mid in [13, 21, 34, 55, ]  # 定义支撑线
                        for close_types in [
                            [  # OpenCloseTypes.CROSS_SUPPORT_MID,
                                # OpenCloseTypes.CROSS_SELF_BL_PLUS,  #
                                # OpenCloseTypes.CROSS_ALL_SUPPORT_MID,
                                OpenCloseTypes.CROSS_DO_BL,  # 对面的布林带穿越
                                # OpenCloseTypes.REVERSE_BREAK_SELF,  # 反向破自身
                            ],
                            # [OpenCloseTypes.CROSS_SELF_BL, ],
                        ]
                        # for stop_loss_type in ['fixed', 'boll_Min1'][:1]
                        # for out_price_type_fixed in ['extreme', 'Close'][1:]  # 固定止损出场判断时用什么的价格
                        # for stop_loss_fixed_p in numpy.arange(0.35, 0.351, 0.025)  # 固定止损比例)
                        ]

# 超参数 取组合后的
hyper_parameter = [dict(**x, **y)
                   for x in all_lens_bars
                   for y in hyper_parameter_base]


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
        use_bl_lens = params.get('min1_bl_lens')
        if isinstance(use_bl_lens, int):
            use_bl_lens = [use_bl_lens]
        min1_bl_dev: float = params.get('min1_bl_dev')
        min1_ma_type: str = params.get('min1_ma_type')
        pos_use_close_type: str = params.get('pos_use_close_type')
        # 进场
        open_types: List[OpenCloseTypes] = params.get('open_types', )  # 开仓条件
        open_need_bars: int = params.get('open_need_bars', )  #
        reopen_interval_bars_for_dev_range: int = params.get('reopen_interval_bars_for_DEV_RANGE', )  #
        reopen_interval_bars_for_support: int = params.get('reopen_interval_bars_for_SUPPORT', )  #
        reopen_need_bars_for_cross_self: int = params.get('reopen_need_bars_for_cross_self', )  #
        # 出场
        reverse_need_bars: int = params.get('reverse_need_bars', )  # 反向破自身需要多少个bar
        close_types: List[OpenCloseTypes] = params.get('close_types', )  # 平仓类型
        close_need_bars_for_small: int = params.get('close_need_bars_for_small', )  #
        support_lens_for_open: List[int] = params.get('support_lens_for_open', )  #
        n_bars_is_support_mid: int = params.get('n_bars_is_support_mid', )  # 定义support
        # stop_loss_type = params.get('stop_loss_type', None)  # 止损判断时用什么的价格
        # out_price_type_fixed = params.get('out_price_type_fixed', )  # 固定止损出场判断时用什么的价格
        # stop_loss_fixed_p = params.get('stop_loss_fixed_p', None)  # 固定止损比例

        # 策略里的交易参数
        trade_p: TradeParams = self.trade_params  # 交易参数
        # 当前处理的bar的index
        curr_index = used_data['min1'].index[-1]

        # 上一轮记录的状态
        if 'prior_pos' not in working_status:
            working_status['prior_pos'] = {}
        prior_bs: Dict[int, BlStatus] = working_status['prior_pos'].copy()
        # 这轮状态
        curr_bs: Dict[int, BlStatus] = {}

        # 一次性遍历完全部length，产生各length信号
        for len_ in fibonacci_list:
            bl_name: str = f'{min1_ma_type}_{len_}'
            # 当前length的布林带数据
            bl_mid = curr_min1[f'boll_mid_{bl_name}']
            bl_std = curr_min1[f'boll_std_{bl_name}']
            if math.isnan(bl_mid):
                return working_trades, working_status  # 数据未完全生成，退出这轮

            # 当前bar的状态
            if len_ in prior_bs:
                curr_bs[len_] = prior_bs[len_]  # 拷贝之前状态
            else:
                curr_bs[len_] = BlStatus(  # 第一次
                    len_=len_,
                    dev=min1_bl_dev,  # 偏移
                    )
            # 无论如何，每次都更新的值
            curr_bs[len_].update(
                ref=curr_min1[pos_use_close_type],
                mid=bl_mid,  # 中值
                std=bl_std,  # std
                curr_index=curr_index, )

        # 当前数据 更新到prior
        working_status['prior_pos'] = curr_bs

        # 具体逻辑 =================================================================

        # 更新全局状态
        working_status['main_direction'] = Direction.opposite(curr_bs[use_bl_lens[-1]].is_break_out_bl)

        # 开仓 -------------------------------------------------
        def batch_get_open_direction() -> Dict[int, OpenCloseParams]:

            # todo 触发部分条件的设置 操作

            """
            批量 获取开仓方向
            """
            open_result = {}
            prior_open_interval = curr_index - working_status['prior_opened_index']
            for len_ in use_bl_lens:  # 公用的，一次性全部周期获取完
                cbs: BlStatus = curr_bs[len_]

                # 在布林带外 多策略
                if OpenCloseTypes.IN_BL_OUTSIDE_PLUS in open_types:
                    # 回破小周期 开仓
                    # if (prior_open_interval >= reopen_interval_bars_for_support
                    #         and cbs.is_out_bl_line):
                    if cbs.is_out_bl_line:
                        for s_len in support_lens_for_open:
                            sbs: BlStatus = curr_bs[s_len]  # 布林带
                            s_direction: Direction = sbs.cross_support_line
                            if Direction.is_opposite(s_direction, cbs.price_bl):
                                open_result[len_] = OpenCloseParams(
                                    direction=s_direction,
                                    type=OpenCloseTypes.IN_BL_OUTSIDE_PLUS,
                                    para=cbs,
                                    price=curr_min1.Last, )
                                break
                        if len_ in open_result:
                            continue  # 有信号

                # 回破布林带上下轨 开仓
                if OpenCloseTypes.CROSS_DO_BL in open_types:
                    # if prior_open_interval >= reopen_interval_bars_for_self_bl:
                        c_direction: Direction = cbs.is_break_in_bl
                        if (c_direction != Direction.NONE
                                and cbs.prior_turn_bl_past
                                and cbs.prior_turn_bl_past >= open_need_bars  # 且过了N个bar
                                # and (curr_index - working_status['prior_opened_index']  # 过n个bar后才能重新开
                                #      >= reopen_need_bars_for_cross_self)
                        ):
                            open_result[len_] = OpenCloseParams(
                                direction=c_direction,
                                type=OpenCloseTypes.CROSS_DO_BL,
                                para=cbs,
                                price=curr_min1.Last, )
                            continue

                # 破强支撑均线，开仓
                # if OpenCloseTypes.CROSS_SUPPORT_MID in open_types:
                #     curr_bs
                #     prior_bs

                # # 破布林带上下轨 且回破小周期 开仓
                # if OpenCloseTypes.CROSS_SELF_BL_PLUS in open_types:
                #     small_bs: BlStatus = curr_bs[21]  # 用21试试
                #     small_cross_mid: Direction = small_bs.is_cross_mid
                #
                #     if (cbs.is_out_bl_line  # 在布林带外
                #             and small_cross_mid != Direction.NONE  # small交叉
                #             and small_cross_mid != cbs.p_bl  # 且相反方向
                #             and (small_bs.prior_turn_mid_past  # 且过了N个bar
                #                  >= )):
                #         open_result[len_] = OpenCloseParams(
                #             direction=small_cross_mid,
                #             type=OpenCloseTypes.CROSS_SELF_BL_PLUS,
                #             para=cbs,
                #             price=curr_min1.Last, )
                #         continue

                # 不满足开仓
                open_result[len_] = OpenCloseParams()

            return open_result

        def reverse_break_self_open_signal(trade_: SimuTrade, ) -> OpenCloseParams:
            """
            反向破自身的延续单 开仓
            """
            direction, len_ = trade_.open_para
            cbs: BlStatus = curr_bs[len_]

            if direction == cbs.is_break_in_bl:
                return OpenCloseParams(direction=direction,
                                       type=OpenCloseTypes.REVERSE_BREAK_MAIN_BL,
                                       para=prior_bs[len_],
                                       price=curr_min1.Last, )
            # 不符合
            return OpenCloseParams()  # 不开仓

        def single_get_open_signal(trade_: SimuTrade, ) -> OpenCloseParams:
            """
            单工单 获取开仓信号
            """

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
        open_batch_signals: Dict[int, OpenCloseParams] = batch_get_open_direction()

        # 开仓处理 遍历各工单(未开仓/已开仓未平仓）
        for a_trade in working_trades:
            if a_trade.waiting_open:
                # 反向破自身 的延续单
                if a_trade.waiting_condition == OpenCloseTypes.REVERSE_BREAK_MAIN_BL:
                    open_signal_ = reverse_break_self_open_signal(a_trade)
                    if open_signal_.direction != Direction.NONE:  # 有信号
                        a_trade = _process_open(a_trade, open_signal_)  # 开仓 处理
                # 普通空白单
                else:
                    for len_ in use_bl_lens:
                        if open_batch_signals[len_].is_valid:  # 有信号 开仓
                            a_trade = _process_open(a_trade, open_batch_signals[len_])
                            break
            else:
                # 已开仓的 只更新最大浮盈
                a_trade.record_extreme(high=curr_min1.High, low=curr_min1.Low,
                                       bar_index=curr_index)
            # 每一个工单都保存起来
            after_open_processed_trades.append(a_trade)

        # 平仓  ==============================================================

        def single_get_close_signal(trade_: SimuTrade, ) -> OpenCloseParams:
            """
            单工单 获取平仓信号
            """
            # # 反转平仓==用开仓信号反向进行平仓
            # if 'opposite_direction' in close_types:
            #     # 现在开仓信号与订单的相反  # 用反向开仓的价格平仓
            #     if Direction.is_opposite(open_signal_.direction, trade_.direction):
            #         return OpenCloseParams(direction=trade_.direction,
            #                                type=f'opposite_open',
            #                                para=open_signal_,
            #                                price=open_signal_.price)

            # # 到时间
            # if 'one_p' in close_types:
            #     if curr_index - trade_.open_bar >= trade_.open_para.len_:
            #         return OpenCloseParams(direction=trade_.direction,
            #                                type=f'arrived one p',
            #                                price=curr_min1.Last, )

            # 反向破自身
            cbs: BlStatus = curr_bs[trade_.open_para.len_]  # 工单对应len的布林带
            if OpenCloseTypes.REVERSE_BREAK_MAIN_BL in close_types:
                if Direction.is_opposite(trade_.direction, cbs.is_break_out_bl):
                    # if ((trade_.direction == Direction.SHORT and curr_min1.Last > trade_.open_para.bl_value)
                    #         or (trade_.direction == Direction.LONG and curr_min1.Last < trade_.open_para.bl_value)):
                    # todo 高于成本
                    # if prior_bs[len_].past > reverse_need_bars:
                    return OpenCloseParams(direction=trade_.direction,  # 平开仓方向
                                           type=OpenCloseTypes.REVERSE_BREAK_MAIN_BL,
                                           price=curr_min1.Last,
                                           para=(trade_.direction, trade_.open_para.len_),
                                           )

            # 止损平仓 ==========================================

            # if stop_loss_type and stop_loss_type == 'fixed':  # 固定止损
            #     if trade_.direction == Direction.LONG:
            #         # 固定止损的触发价，==开盘价*（1-回落百分比）
            #         sl_triggle_price = trade_.open_price * (1 - stop_loss_fixed_p / 100)
            #         # 判断用的价格
            #         used_price_ = (curr_min1.Last if out_price_type_fixed == 'Close' else curr_min1.Low)
            #         if used_price_ <= sl_triggle_price:
            #             return OpenCloseParams(direction=trade_.direction,
            #                                    type='fixed_stop_loss',
            #                                    price=(used_price_
            #                                           if out_price_type_fixed == 'Close'
            #                                           else sl_triggle_price), )
            #     elif trade_.direction == Direction.SHORT:  # 平空
            #         sl_triggle_price = trade_.open_price * (1 + stop_loss_fixed_p / 100)
            #         used_price_ = (curr_min1.Last if out_price_type_fixed == 'Close' else curr_min1.High)
            #         if used_price_ >= sl_triggle_price:
            #             return OpenCloseParams(direction=trade_.direction,
            #                                    type='fixed_stop_loss',
            #                                    price=(used_price_
            #                                           if out_price_type_fixed == 'Close'
            #                                           else sl_triggle_price), )

            # 都不符合
            return OpenCloseParams()  # 不平仓

        def batch_get_close_signal() -> Dict[int, OpenCloseParams]:
            """
            批量 获取平仓信号
            """
            result = {}
            for len_ in use_bl_lens:  # 所有周期的都检查下 有没有平仓信号产生
                cbs: BlStatus = curr_bs[len_]

                # 回破布林带上下轨 平仓
                if OpenCloseTypes.CROSS_DO_BL in close_types:
                    # 突破（开仓）的方向
                    if (cbs.is_break_in_bl != Direction.NONE):
                        result[len_] = OpenCloseParams(
                            direction=(Direction.LONG
                                       if cbs.is_break_in_bl == Direction.SHORT
                                       else Direction.SHORT),
                            type=OpenCloseTypes.CROSS_DO_BL,
                            para=cbs,
                            price=curr_min1.Last, )
                        continue

                # # 破布林带上下轨后，回破小周期 平仓
                # if OpenCloseTypes.CROSS_SUPPORT_MID in close_types:
                #     main_direction: Direction = working_status['main_direction']
                #     support_bs: BlStatus = curr_bs[close_cross_support_len]
                #
                #     if main_direction:
                #         # 同向破了有效支撑线
                #         if ((support_bs.is_cross_mid == main_direction
                #              and support_bs.prior_turn_mid_past >= n_bars_is_support_mid)
                #                 # 支撑线无效 且 回到布林带内
                #                 or (support_bs.mid_past < n_bars_is_support_mid
                #                     and not cbs.is_out_bl_line)):
                #             result[len_] = OpenCloseParams(
                #                 direction=Direction.opposite(main_direction),
                #                 type=OpenCloseTypes.CROSS_SUPPORT_MID,
                #                 para=cbs,
                #                 price=curr_min1.Last, )
                #             continue
                #
                # # 破布林带上下轨后，突破一组小周期布林带，平仓
                # if OpenCloseTypes.CROSS_ALL_SUPPORT_MID in close_types:
                #     pass

                # 不满足
                result[len_] = OpenCloseParams()

            return result

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
        close_signals = batch_get_close_signal()

        for a_trade in after_open_processed_trades:
            if a_trade.waiting_close:
                # 先判断公用的
                close_signal_ = close_signals[a_trade.open_para.len_]
                if close_signal_.direction == a_trade.direction:  # 是这个工单需要的平仓方向
                    a_trade = _process_close(a_trade, close_signal_)
                else:
                    # 私有的
                    close_signal_ = single_get_close_signal(a_trade)
                    if close_signal_.direction == a_trade.direction:
                        a_trade = _process_close(a_trade, close_signal_)
            # 收集回来
            result.append(a_trade)

        return result, working_status  # body()
