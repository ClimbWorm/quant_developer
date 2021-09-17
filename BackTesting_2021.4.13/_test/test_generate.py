#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
#  
# Author: Tayii
# Data : 2021/1/26
# ----------------------------------------------------
from config import BACK_TESTING_RESULT_DIR
from datahub.generate import *
import talib
from dc.source_data import *
from datahub.generate import *
import pandas as pd
from os import path

pd.set_option('display.max_columns', None)

plan_name = 'strategy_tri_ma_min1_0227_ESH21'
filepath = path.join(BACK_TESTING_RESULT_DIR, f'{plan_name}', f'{0}.csv')


# def test_annual_std():
#     print(df.columns)
#     print(df)


def test_Performance():
    df = get_back_testing_result_data(filepath)
    print('aa')
    per = Performance('Performance', df)
    # print(per.df_all)
    # print(per.df_long)
    # print(3,per.df_short)

    # print(per.calc_trades())
    print(per.calc_earnings())
    # print(per.calc_duration())
    'day_earnings \n', per.calc_day_earnings()
    # print(per.df_all)
    #
    # print(2, per.day_pnl)
    # print(per.day_earnings_rate)
    # print(per.annual_std)

    # per.show_result()
    # print(per.day_earnings)


# def test_annual_return():
#     assert False



def test_run():
    df = get_source_data_from_config('YMH21_1_SC')[:100]

    q_from_caller = multiprocessing.Queue(2)
    q_to_caller = multiprocessing.Queue(2)
    barg = BarGenerator(q_from_caller, q_to_caller,
                        data_set_name='min5',
                        strategy_import_path='strategies.strategy_tri_ma')
    barg.start()
    print(123)
    while True:
        print('接收...')
        msg = q_to_caller.get()
        print(msg)
        time.sleep(3)
