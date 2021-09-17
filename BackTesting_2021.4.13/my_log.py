#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 自定义日志
#
# Author: Tayii
# Data : 2020/7/9 16:54
# ----------------------------------------------------

import logging
from logging import handlers
import sys
from time import strftime
from os import path

from config import LOG_SETTING


# ----------------------------------------------------
# 自定义日志，在logging基础上封装
# ----------------------------------------------------
class MyLog(object):
    """
    对外接口：
        to_console(self, message, level=logging.DEBUG)
            输出到控制台
        to_ui(self, message, level=logging.INFO)
            输出到前端
        to_file(self, message, level=logging.INFO)
            输出到本地文件
        日志输出到控制台，可按level单独使用以下接口：
        debug(self, message)
            输出到控制台 level=DEBUG
        info(self, message)
            输出到控制台(默认同步保存到本地文件) level=INFO
        warning(self, message)
            输出到控制台 level=WARNING
        error(self, message, to_file=True)
            输出到控制台(默认同步保存到本地文件) level=INFO
        critical(self, message, to_file=True)
            输出到控制台(默认同步保存到本地文件) level=INFO
    """

    def __init__(self, name: str = None) -> None:
        """
        Args:
            name: 日志实例名
        """
        # 为不同的输出，设置不同的logger
        # 因为MyLog要全局共享，频繁加载卸载handle可能影响效率，多线程调用可能会出错
        self.name = name or __name__
        self.__levels = LOG_SETTING['level']
        self.__logger_c = None
        self.fmt = '%(message)s      ------  %(asctime)s [%(levelname)s]'
        logging.basicConfig(format=self.fmt,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=self.__levels['main'],
                            stream=sys.stdout,
                            )  # 默认输出到console

    def console(self, message, level=logging.DEBUG, exc_info=False):
        """
        输出到控制台
        Args:
            message: 消息
            level: 日志级别，默认DEBUG
            exc_info: 是否需要输出异常信息
        """
        if not self.__logger_c or len(self.__logger_c.handlers) == 0:
            self.__logger_c = logging.getLogger(f'{self.name}')
            self.__logger_c.propagate = False  # propagate的作用, 它把子代的所有record都发送给了父代, 循环往复, 最终到达root
            sh = logging.StreamHandler(sys.stdout)  # console handel
            sh.setLevel(self.__levels['console'])
            sh.setFormatter(logging.Formatter(self.fmt))
            self.__logger_c.addHandler(sh)

        self.__logger_c.log(level, message, exc_info=exc_info)

    def debug(self, message):
        """输出到控制台 level=DEBUG"""
        self.console(message, logging.DEBUG)

    def info(self, message):
        """输出到控制台 level=INFO"""
        self.console(message, logging.INFO)

    def warning(self, message):
        """输出到控制台 level=WARNING"""
        self.console(f'! {message}', logging.WARNING)

    def error(self, message, exc_info=True):
        """输出到控制台 level=ERROR"""
        self.console(f'!!! {message}', logging.ERROR, exc_info=exc_info)

    def critical(self, message, exc_info=True):
        """输出到控制台 level=CRITICAL"""
        self.console(f'!!! {message}', logging.CRITICAL, exc_info=exc_info)

    # self.__f_level = LOG_SETTING['level']['file']
    # self.__ui_level = LOG_SETTING['level']['ui']
    # print(1, logging.getLogger("requests"))
    #
    # # requests模块请求日志level调整
    # logging.getLogger("requests").setLevel(logging.WARNING)

    @classmethod
    def to_ui(cls, message, level=logging.INFO):
        """
        输出到前端
        Args:
            message: 消息
            level: 日志级别，默认INFO
        """
        logger_ui = None
        print(f'log: to_ui还未实现...{level}', )

    @classmethod
    def to_file(cls, message, level=logging.ERROR, set_level=None):
        """
        输出到本地文件
        TODO 操作完是否需要关闭文件?
        Args:
            message: 消息
            level: 保存到文件的日志级别
        """
        filename = path.join(LOG_SETTING['file_dir'],
                             f'{strftime("%Y_%m_%d")}.log')  # log文件名
        fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s消息: %(message)s'

        logger_f = logging.getLogger(f'to_file')

        if set_level:
            for handler in logger_f.handlers:
                logger_f.removeHandler(handler)  # 清空 重置

        if len(logger_f.handlers) == 0:  # 未初始化过
            logger_f.propagate = False
            # fh = logging.FileHandler(self.filename, 'a', encoding='utf-8')  # file handel
            # 日志文件最大20M,超过时就会覆盖之前的
            # fh = handlers.RotatingFileHandler(self.filename,   # file handel
            #                                           maxBytes=20 * 1024 * 1024,
            #                                           backupCount=5, encoding='utf-8')
            # 定义一个一天换一次日志文件的渠道（每天生成一个新的日志文件），
            # 最多保留5个旧的日志文件，也就是说保留最新5天的日志文件
            fh = handlers.TimedRotatingFileHandler(filename,  # file handel
                                                   when='D', interval=1,
                                                   backupCount=5,  # 0==不会自动删除掉日志
                                                   encoding='utf-8')
            try:
                set_level = set_level or LOG_SETTING['level']['file']
            except:
                set_level = logging.WARNING
            finally:
                fh.setLevel(set_level)

            fh.setFormatter(logging.Formatter(fmt))
            logger_f.addHandler(fh)

        logger_f.log(level, message)


# 全局共享的 自定义日志记录器
ilog = MyLog(__name__)

__all__ = [
    'MyLog',
    'ilog',
]

if __name__ == '__main__':
    pass