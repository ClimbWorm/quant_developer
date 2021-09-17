#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 公用 底层接口
# Author: Tayii
# Data : 2020/12/01
# ----------------------------------------------------
import sys
from time import strftime
from decimal import Decimal
from math import floor, ceil
import time
from datetime import datetime
from functools import wraps
import traceback
import threading


# ----------------------------------------------------
# 装饰器 #

def timeit(func):
    """
    @装饰器 计算func消耗的时间
    PS: 多装饰器时 放在外层
    Args:
        func: 被装饰的函数
    """

    @wraps(func)  # 把原始函数的__name__等属性复制到wrapper()函数中
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        cost = time.time() - start
        print('.' * 30 + '【{}】  cost {:.2f}\'S'.format(func.__name__, cost))
        return ret

    return wrapper


def catch_except(info=None, ret=None):
    """
    @装饰器 捕获异常
    Args:
        info: 额外的打印信息(可选)
        ret: 指定异常时候的返回值
    Returns:
        正常：返回func函数的正常返回
        异常：返回ret的值 并打印错误信息
    """

    def decorated(func):
        @wraps(func)  # 把原函数的元信息拷贝到装饰器里面的 func函数中
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(f'\n{info} error in "{func.__name__}()" : {e} {exc_tb.tb_frame}')
                message = """\n
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv Start vvv
[ Time: {}, Error Func: {} ]
{}: {}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ -End- ^^^
                        """.format(time_now(), e, func.__name__, traceback.format_exc())
                print(message)
                return ret  # 入参

        return wrapper

    return decorated


@catch_except()
def timestamp2str(timestamp: float, add_hour: int = 0):
    # 10位时间戳转换为字符串
    d = timestamp2datetime(timestamp, add_hour)
    return d.strftime("%Y-%m-%d %H:%M:%S")


def time_now():
    return timestamp2str(time.time())  # 现在时间: 字符串


def show_active_threads():
    """
    @ 查看当前存活的线程们
    """
    print('\n\n★当前活跃的线程数 ={}  它们是：'.format(threading.active_count()))
    ths = threading.enumerate()  # 以列表形式返回当前所有存活的 Thread 对象。
    for i in range(len(ths)):
        try:
            print('--', ths[i])
        except Exception as e:
            print('-err-', ths[i], e)


def round_to(value: float, target: float) -> float:
    """
    Round price to price tick value.
    """
    value = Decimal(str(value))
    target = Decimal(str(target))
    rounded = float(int(round(value / target)) * target)
    return rounded


def floor_to(value: float, target: float) -> float:
    """
    Similar to math.floor function, but to target float number.
    """
    value = Decimal(str(value))
    target = Decimal(str(target))
    result = float(int(floor(value / target)) * target)
    return result


def ceil_to(value: float, target: float) -> float:
    """
    Similar to math.ceil function, but to target float number.
    """
    value = Decimal(str(value))
    target = Decimal(str(target))
    result = float(int(ceil(value / target)) * target)
    return result


# # 将时间差转换为datetime对象
# date = datetime.datetime.fromtimestamp(now)
# 1440751417.283 --> '2015-08-28 16:43:37.283'
@catch_except()
def timestamp2datetime(timestamp: float, add_hour: int = 0):
    """
    时间戳转换成字符串日期时间（秒，整数是10位）
    Args:
        timestamp: 时间戳
        add_hour: 增加时间（小时，不同时区用）
    Returns:
        datetime
        None: 异常 错误
    """
    assert isinstance(add_hour, int)
    # 先把时间戳转换成秒级（10位）
    if timestamp <= 0:
        return
    elif timestamp < 10 ** 10:
        pass
    elif timestamp < 10 ** 13:
        timestamp /= 10 ** 3
    elif timestamp < 10 ** 16:
        timestamp /= 10 ** 6
    else:
        return

    # 格式化时间戳为struct_time对象，接着格式化输出
    timestamp = timestamp + 3600 * add_hour  # 加时差n个小时
    # loc_time = time.localtime(timestamp)
    # string = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)
    return datetime.fromtimestamp(timestamp)


@catch_except()
def timestamp2str(timestamp: float, add_hour: int = 0):
    # 10位时间戳转换为字符串
    d = timestamp2datetime(timestamp, add_hour)
    return d.strftime("%Y-%m-%d %H:%M:%S")


@catch_except()
def str2timestamp(s, ms=True):
    # 字符串转时间戳  ms是否有毫秒
    if ms:
        d = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    else:
        d = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = time.mktime(t) + float(d.microsecond) / 1000000
    return timeStamp

