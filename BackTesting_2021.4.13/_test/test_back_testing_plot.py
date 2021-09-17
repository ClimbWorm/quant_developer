#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 
# 
# Author: Tayii
# Data : 2021/2/8
# ----------------------------------------------------
from config import BACK_TESTING_RESULT_DIR
import pandas as pd
from os import path

from dc.source_data import *
from datahub.generate import *
from  plot.back_testing_plot import *


pd.set_option('display.max_columns', None)

plan_name = 'back_testing_20210202_triMa'
filepath = path.join(BACK_TESTING_RESULT_DIR, f'{plan_name}', f'{0}.csv')


def test_plot_day_pnl():
    df = get_back_testing_result_data(filepath)
    # print(df)
    perf = Performance('Performance', df)
    # perf.show_result()

    btc = PlotBtSingleChart(perf)
    print(f'plot_day_pnl-- \n', btc.plot_day_pnl())


