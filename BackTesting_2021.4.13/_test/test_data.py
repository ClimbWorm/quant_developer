#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
#  
# Author: Tayii
# Data : 2021/1/27
# ----------------------------------------------------

import logging
import os
from typing import Any, Dict, Optional, List
import time
import datetime
from logging import INFO
import threading
from multiprocessing import Queue, Manager, Process, Pool, cpu_count
from abc import ABC, abstractmethod
import pandas as pd
from importlib import import_module
from os import path
from engine.data import *
from multiprocessing import Queue

q_from_caller = multiprocessing.Queue(2)
q_to_caller = multiprocessing.Queue(2)
de = DataEngine(q_from_caller, q_to_caller,
                    data_set_name='min5',
                    strategy_import_path='strategies.strategy_tri_ma')


def test_run():
    print(de)
    de.start()

    while True:
        print('接收de...')
        msg = q_to_caller.get()
        print(msg)
        time.sleep(3)
