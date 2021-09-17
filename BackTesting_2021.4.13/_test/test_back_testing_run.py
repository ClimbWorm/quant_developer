#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
#  
# Author: Tayii
# Data : 2021/1/20
# ----------------------------------------------------

from backtester.back_testing_run import *

from dc.source_data import *
from datahub.generate import *
df = get_source_data_from_config('sc', 'YMH21')

def test_BackTester():
    print(df)

    paras = {"rsi": 70, }
    BackTester.back_testing_run(df, paras)