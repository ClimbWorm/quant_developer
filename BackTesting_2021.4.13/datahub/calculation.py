#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
#  统计及计算
# Author: Tayii
# Data : 2021/1/25
# ----------------------------------------------------
import pandas as pd



class Calc(object):

    @classmethod
    def calculate_result(cls, result: pd.DataFrame):
        """
        计算策略盈亏情况
        基于收盘价、当日持仓量、合约规模、滑点、手续费率等计算总盈亏与净盈亏
            浮动盈亏 = 持仓量 *（当日收盘价 - 昨日收盘价）* 合约单位
            实际盈亏 = 持仓变化量 * （当时收盘价 - 开仓成交价）* 合约单位
            总盈亏 = 浮动盈亏 + 实际盈亏
            总净盈亏 = 总盈亏 - 总手续费 - 总滑点
        """
        cls.log("开始计算策略盈亏情况")
        # 按日统计
        day_grouped = result.groupby(pd.DatetimeIndex(result['open_datetime']).date)

        profit_sum = profit_p_sum = profit_total_sum = 0
        if result is not None and len(result) > 0:
            profit_sum = result['profit'].sum()
            profit_p_sum = result['profit_p'].sum()
            profit_total_sum = result['profit_total'].sum()

        print(f'共{len(result)}笔： profit_sum={profit_sum} '
              f'profit_p_sum={profit_p_sum:.2f}% profit_total_sum={profit_total_sum}')

        # if not self.trades:
        #     self.log("成交记录为空，无法计算")
        #     return
        #
        # # Add trade data into daily reuslt.
        # for trade in self.trades.values():
        #     d = trade.datetime.date()
        #     daily_result = self.daily_results[d]
        #     daily_result.add_trade(trade)
        #
        # # Calculate daily result by iteration.
        # pre_close = 0
        # start_pos = 0
        #
        # for daily_result in self.daily_results.values():
        #     daily_result.calculate_pnl(
        #         pre_close,
        #         start_pos,
        #         self.size,
        #         self.rate,
        #         self.slippage,
        #         self.inverse
        #     )
        #
        #     pre_close = daily_result.close_price
        #     start_pos = daily_result.end_pos
        #
        # # Generate dataframe
        # results = defaultdict(list)
        #
        # for daily_result in self.daily_results.values():
        #     for key, value in daily_result.__dict__.items():
        #         results[key].append(value)
        #
        # self.daily_df = pd.DataFrame.from_dict(results).set_index("date")
        #
        # self.output("逐日盯市盈亏计算完成")
        # return self.daily_df