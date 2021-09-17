#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 回测 运行

# Author: Tayii
# Data : 2021/1/18
# ----------------------------------------------------
import random
import time
from pprint import pprint

import pandas as pd
from os import path
from typing import Optional, List, Dict

from backtester.template import BackTestingDataSetting
from config import BACK_TESTING_RESULT_DIR
from constant import Direction
from datahub.generate import Performance
from dc.config import DataSourceConfig
from my_log import ilog
import logging
from datahub.fmt import SimuTrade
from importlib import import_module
from constant import OpenMax
from strategies.template import TradeParams, StrategyTemplate, OpenCloseTypes
from utility import catch_except


class BackTester(object):
    """
    回测器
    """

    @classmethod
    @catch_except()
    def back_testing_run(cls,
                         plan_name: str,  # 本次回测计划 名
                         data_source: DataSourceConfig,  # 数据源配置
                         strategy_import_path: str,  # 选择的策略 包完整路径
                         ok_data: Dict[str, pd.DataFrame],  # 处理好的数据 全部
                         n: int,  # 回测用的 第n组超参数
                         save_trade_result: bool = True,  # 是否保存回测交易结果
                         ) -> Optional[str]:
        """
        回测 主运行（基于某一组参数）
        """
        # 生成策略实例
        _s = import_module(strategy_import_path).Strategy
        strategy: StrategyTemplate = _s(
            name=plan_name,  # 此处支持一次回测多个数据源
            data_source=data_source,
        )

        parameters: dict = (strategy.hyper_parameter[n]  # 这组超参数
                            if len(strategy.hyper_parameter) else {})
        cls.log(f'独立进程 开始... \n  '
                f'- 回测第{n}组，超参数={parameters if len(parameters) else "无"}')

        # 交易&仓位管理
        finished_trades: Optional[pd.DataFrame] = None  # 已结束的订单们
        working_trades = []  # 进行中订单们==已开仓未平仓
        working_status = {
            'prior_opened_index': -100,  # 上一次开仓是哪个bar
            'main_direction': Direction.NONE,  # 主交易方向
        }  # 为每一轮提供上一轮的各种数据状态

        def register_a_new_trade() -> None:
            """注册一个新工单（未开仓）"""
            _new_trade = SimuTrade(
                usd_per_tick=trade_p.usd_per_tick,  # 每个tick价值n美金
                tick_size=trade_p.tick_size,
                fee_rate=trade_p.fee_rate,
                fee_amount=trade_p.fee_amount,
                slippage=trade_p.slippage, )
            # 添加到列表
            working_trades.append(_new_trade)

        def trade_unregister(trade_: SimuTrade) -> None:
            """工单注销"""
            for i in range(len(working_trades)):
                if working_trades[i] == trade_:
                    working_trades.pop(i)
                    return

        def trade_register(trade_: SimuTrade) -> None:
            """工单注册 可能是其他地方新开的订单 加入到列表"""
            #  避免多空白订单
            if (not trade_.is_black_trade) or (trade_ not in working_trades):
                working_trades.append(trade_)

        trade_p: TradeParams = strategy.trade_params  # 交易参数
        total_money: float = trade_p.init_money  # 总资产

        need_bars: int = strategy.need_bars_once  # 一次切片回测需要最近几个bar数据
        # 逐一回测 所有切片（去掉前面n个）
        size = len(list(ok_data.values())[0])
        _mark_size: int = int(size / random.uniform(3.5, 7))  # 显示用
        begin = time.time()
        bar_opened_index = -1  # 已开单的bar index  针对多单

        for index in range(need_bars - 1, size):
            if index % _mark_size == 0:
                cls.log(f'第{n}组 回测了{index / size:.1%} 耗时：{time.time() - begin:.1f}秒')

            # 查询未平仓订单  # 决定开平仓逻辑
            if trade_p.open_max == OpenMax.LONG_1_OR_SHORT_1:
                if len(working_trades) == 0:  # []==空仓中
                    # 新增一笔交易 初始状态=None
                    register_a_new_trade()
            elif trade_p.open_max == OpenMax.LONG_n_OR_SHORT_n:

                def has_a_black_trade() -> bool:
                    for trade_ in working_trades:
                        if trade_.is_black_trade:
                            return True
                    return False

                if index > bar_opened_index and not has_a_black_trade():
                    # 没有空白单 新建一个
                    register_a_new_trade()  # 新增一笔交易==允许开新单
            else:
                raise Exception(f'{trade_p.open_max} 没有实现')

            # 当前轮到的切片的回测 用n行数据
            used_data: Dict[str, pd.DataFrame] = {}
            for name, df in ok_data.items():
                used_data[name] = df[index + 1 - need_bars:index + 1]  # .reset_index(drop=True)

            # 执行一次切片回测  返回：工单们，各种需记录的实时状态
            # ret_trades: List[SimuTrade], working_status: Dict
            ret_trades, working_status = strategy.body(
                working_trades,  # 送入全部的 待开仓和已开仓订单们
                working_status,  # 上一轮记录的各种状态
                used_data,
                parameters,
            )

            # todo 计算浮盈 跨日 需记录下持仓状态和K线状态

            # 对回测的所有trade进行处理
            working_trades = []  # 先清空
            for trade in ret_trades:
                if trade.finished:
                    # 一单交易已完结，添加到result
                    if finished_trades is None:
                        finished_trades = pd.DataFrame(trade.result_as_series()).T
                    else:
                        finished_trades = finished_trades.append(
                            trade.result_as_series(), ignore_index=True)

                    # trade结束 后续处理
                    total_money += trade.earnings  # 更新现有资产

                    # # 如果反向 则新开一个工单 并开仓
                    # if trade.close_type == 'opposite_direction':
                    #     new_opened_trade: SimuTrade = SimuTrade(
                    #         usd_per_tick=trade_p.usd_per_tick,
                    #         tick_size=trade_p.tick_size,
                    #         fee_rate=trade_p.fee_rate,
                    #         fee_amount=trade_p.fee_amount,
                    #         slippage=trade_p.slippage,
                    #     )
                    #     direction_ = (Direction.LONG if trade.open_direction == Direction.SHORT
                    #                   else Direction.SHORT)
                    #     new_opened_trade.set_open(direction_, trade.close_bar, price=trade.close_price,
                    #                               amount=trade_p.once_lots, datetime_=trade.close_datetime,
                    #                               open_type='opposite_direction',)
                    #     # 加入列表
                    #     trade_register(new_opened_trade)

                    # 结束的 工作列表里剔除
                    trade_unregister(trade)
                    # 如果是临时性的，重新开一个工单
                    if trade.close_type == OpenCloseTypes.REVERSE_BREAK_MAIN_BL:
                        # 反身突破 新开空白单并加入列表
                        new_opened_trade: SimuTrade = SimuTrade(
                            usd_per_tick=trade_p.usd_per_tick,
                            tick_size=trade_p.tick_size,
                            fee_rate=trade_p.fee_rate,
                            fee_amount=trade_p.fee_amount,
                            slippage=trade_p.slippage,
                            waiting_condition=OpenCloseTypes.REVERSE_BREAK_MAIN_BL,
                            open_para=trade.close_para,  # 原来方向和length
                        )
                        trade_register(new_opened_trade)
                else:
                    trade_register(trade)  # 放回未结束的（可能有新的）

        # 所有回测已完成
        cls.log(f'\n{plan_name}: 第{n}组 回测已完成, 最终资产：{total_money} \n'
                f'参数:  \n')
        pprint(parameters)

        if finished_trades is None or len(finished_trades) == 0:
            cls.log(f'第{n}组 回测结果：没有一单成交')
        else:
            print(f'第{n}组 回测结果(最多显示最后5单):\n', finished_trades.tail(5), )
            if len(working_trades) and working_trades[0].waiting_close:
                print(f'\n  最终还未平仓{len(working_trades)}订单：\n')

        # 本次回测 总的回测结果
        param_str = ' '.join(f'{x}' for x in parameters.values())  # 参数

        # 回测结果  写入本地文件
        trade_filepath: str = ''
        if save_trade_result and finished_trades is not None:
            trade_filepath = path.join(BACK_TESTING_RESULT_DIR,
                                       f'{plan_name}', f'{n}.csv')
            finished_trades.to_csv(trade_filepath)
            cls.log(f'{plan_name} 第{n}组参数 回测结果 已保存到:{trade_filepath}')

        # 计算结果
        perf = Performance(name=f'Performance {n}',
                           init_asset=trade_p.init_money,
                           df_all=finished_trades, )
        # perf.show_result()

        # 返回回测结果给回调函数
        # 超参数序号  超参数  (days total_earnings)  文件路径
        return f'{n}  {param_str} {trade_filepath}'

    @classmethod
    def log(cls, msg, level=logging.DEBUG, exc_info=False) -> None:
        ilog.console(f'BackTester {msg}', level=level, exc_info=exc_info)


if __name__ == '__main__':
    print(BackTester.back_testing_run)
